"""本機資料庫 market.db（SQLite）＋ 自選股清單 config.json。

所有資料都存在這台電腦，每天只補抓新的部分：
  prices      日 K（已剔除休市日）
  inst        三大法人買賣超（台股）
  margin      融資融券（台股）
  divs        除權息
  meta        股票名稱、上市/上櫃、最後更新時間
  rule_rec    規則模型每天的回溯預測（只算新的日期）
  ml_oos      機器學習每天的樣本外預測
  model_state 每檔每週期的模型狀態（冠軍、成績、特徵重要度、校正權重）
  forecast    今日最新預測（每個模型一筆）
  live        實際預測紀錄（每天每模型一筆，過去不改寫）
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pandas as pd

import data as D

APP_DIR = Path(__file__).parent
DB = APP_DIR / "market.db"
CONFIG = APP_DIR / "config.json"
LOCK = threading.Lock()          # 同一時間只跑一個更新

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (code TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS inst (code TEXT, date TEXT, foreign_ REAL, trust REAL, dealer REAL, total REAL,
  PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS margin (code TEXT, date TEXT, mbal REAL, mchg REAL, mlimit REAL, sbal REAL, schg REAL,
  PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS divs (code TEXT, date TEXT, amount REAL, PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS meta (code TEXT PRIMARY KEY, name TEXT, mtype TEXT, last_update TEXT);
CREATE TABLE IF NOT EXISTS rule_rec (code TEXT, date TEXT, data TEXT, PRIMARY KEY (code, date));
CREATE TABLE IF NOT EXISTS ml_oos (code TEXT, date TEXT, h INTEGER, ridge REAL, gbdt REAL, ml REAL,
  PRIMARY KEY (code, date, h));
CREATE TABLE IF NOT EXISTS model_state (code TEXT, h INTEGER, data TEXT, updated TEXT, PRIMARY KEY (code, h));
CREATE TABLE IF NOT EXISTS forecast (code TEXT, model TEXT, date TEXT, data TEXT, updated TEXT,
  PRIMARY KEY (code, model));
CREATE TABLE IF NOT EXISTS live (code TEXT, date TEXT, model TEXT, close REAL, atr REAL,
  p5 REAL, lo5 REAL, hi5 REAL, p20 REAL, lo20 REAL, hi20 REAL, dir5 INTEGER, dir20 INTEGER, logged_at TEXT,
  PRIMARY KEY (code, date, model));
"""
_inited = False


def conn() -> sqlite3.Connection:
    global _inited
    c = sqlite3.connect(DB, timeout=60)
    if not _inited:
        c.executescript(SCHEMA)
        _inited = True
    return c


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _dstr(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


# ================================================================ 自選股清單
DEFAULT_LISTS = {"我的自選": ["2330", "2454", "0050", "NVDA", "TSM", "AAPL"]}


def load_cfg() -> dict:
    cfg = {}
    if CONFIG.exists():
        import re
        text = CONFIG.read_text(encoding="utf-8-sig")              # 容忍 BOM
        text = re.sub(r",\s*([}\]])", r"\1", text)               # 容忍最後一項多一個逗號
        try:
            cfg = json.loads(text)
        except Exception as e:
            print(f"⚠️ config.json 格式有誤，暫用預設清單：{e}")
    if "watchlists" not in cfg:                       # 舊版只有一組 watchlist
        cfg["watchlists"] = {"我的自選": cfg.get("watchlist") or DEFAULT_LISTS["我的自選"]}
    if not cfg["watchlists"]:
        cfg["watchlists"] = dict(DEFAULT_LISTS)
    cfg["watchlists"] = {k: list(dict.fromkeys(str(x).strip().upper() for x in v if str(x).strip()))
                         for k, v in cfg["watchlists"].items()}              # 去除重複、空白
    cfg.setdefault("finmind_token", "")
    cfg.setdefault("model_mode", "auto")             # auto / rule / calib / ml
    cfg.setdefault("auto_update", True)
    return cfg


def save_cfg(cfg: dict) -> None:
    cfg = dict(cfg)
    cfg.pop("watchlist", None)
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def all_codes(cfg: dict) -> list[str]:
    seen = []
    for lst in cfg["watchlists"].values():
        for c in lst:
            if c not in seen:
                seen.append(c)
    return seen


# ================================================================ 價格
def load_prices(code: str) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query("SELECT date, open, high, low, close, volume FROM prices WHERE code=? ORDER BY date",
                               c, params=(code,))
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.to_datetime(df.pop("date"))
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return D.clean_sessions(df, code)


def _write_prices(code: str, df: pd.DataFrame, replace_all: bool = False) -> None:
    rows = [(code, _dstr(d), float(r.Open), float(r.High), float(r.Low), float(r.Close), float(r.Volume or 0))
            for d, r in df.iterrows()]
    with conn() as c:
        if replace_all:
            c.execute("DELETE FROM prices WHERE code=?", (code,))
        c.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)", rows)


def update_prices(code: str, mtype: str | None = None, token: str = "") -> int:
    """增量更新；若發現歷史價格被調整（分割/減資），整段重抓。回傳新增筆數。"""
    old = load_prices(code)
    if old.empty:
        new = D.price_history(code, "max", mtype, token)
        if new.empty:
            return 0
        _write_prices(code, new, replace_all=True)
        return len(new)
    gap = (pd.Timestamp.today().normalize() - old.index[-1]).days
    period = "1mo" if gap <= 20 else "1y" if gap < 330 else "max"
    new = D.price_history(code, period, mtype, token)
    if new.empty:
        return 0
    overlap = new.index.intersection(old.index)[:-1]          # 最後一天可能是盤中資料，不比對
    if len(overlap):
        diff = (new.loc[overlap, "Close"] / old.loc[overlap, "Close"] - 1).abs().max()
        if diff > 0.01:
            full = D.price_history(code, "max", mtype, token)
            if not full.empty:
                _write_prices(code, full, replace_all=True)
                return len(full)
    _write_prices(code, new)
    return int((~new.index.isin(old.index)).sum())


# ================================================================ 籌碼 / 股利
def update_inst(code: str, token: str = "") -> int:
    if not D.is_tw(code):
        return 0
    with conn() as c:
        last = c.execute("SELECT MAX(date) FROM inst WHERE code=?", (code,)).fetchone()[0]
    days = 1100 if not last else max(10, (pd.Timestamp.today() - pd.Timestamp(last)).days + 7)
    df = D.institutional(code, days, token)
    if df.empty:
        return 0
    rows = [(code, _dstr(d), float(r["外資"]), float(r["投信"]), float(r["自營商"]), float(r["合計"]))
            for d, r in df.iterrows()]
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO inst VALUES (?,?,?,?,?,?)", rows)
    return len(rows)


def load_inst(code: str) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query("SELECT date, foreign_, trust, dealer, total FROM inst WHERE code=? ORDER BY date",
                               c, params=(code,))
    if df.empty:
        return pd.DataFrame(columns=["外資", "投信", "自營商", "合計"])
    df.index = pd.to_datetime(df.pop("date"))
    df.columns = ["外資", "投信", "自營商", "合計"]
    return df


def update_margin(code: str, token: str = "") -> int:
    if not D.is_tw(code):
        return 0
    with conn() as c:
        last = c.execute("SELECT MAX(date) FROM margin WHERE code=?", (code,)).fetchone()[0]
    days = 1100 if not last else max(10, (pd.Timestamp.today() - pd.Timestamp(last)).days + 7)
    df = D.margin(code, days, token)
    if df.empty:
        return 0
    f = lambda v: None if pd.isna(v) else float(v)
    rows = [(code, _dstr(d), f(r["融資餘額"]), f(r["融資增減"]), f(r["融資限額"]), f(r["融券餘額"]), f(r["融券增減"]))
            for d, r in df.iterrows()]
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO margin VALUES (?,?,?,?,?,?,?)", rows)
    return len(rows)


def load_margin(code: str) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query("SELECT date, mbal, mchg, mlimit, sbal, schg FROM margin WHERE code=? ORDER BY date",
                               c, params=(code,))
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.pop("date"))
    df.columns = ["融資餘額", "融資增減", "融資限額", "融券餘額", "融券增減"]
    df["券資比%"] = (df["融券餘額"] / df["融資餘額"].replace(0, np.nan) * 100).round(2)
    df["融資使用率%"] = (df["融資餘額"] / df["融資限額"].replace(0, np.nan) * 100).round(2)
    return df


def update_divs(code: str, mtype: str | None = None) -> int:
    s = D.dividends(code, mtype)
    if s.empty:
        return 0
    with conn() as c:
        c.execute("DELETE FROM divs WHERE code=?", (code,))
        c.executemany("INSERT OR REPLACE INTO divs VALUES (?,?,?)",
                      [(code, _dstr(d), float(v)) for d, v in s.items()])
    return len(s)


def load_divs(code: str) -> pd.Series:
    with conn() as c:
        df = pd.read_sql_query("SELECT date, amount FROM divs WHERE code=? ORDER BY date", c, params=(code,))
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df["amount"].values, index=pd.to_datetime(df["date"]))


# ================================================================ meta
def get_meta(code: str) -> dict:
    with conn() as c:
        r = c.execute("SELECT name, mtype, last_update FROM meta WHERE code=?", (code,)).fetchone()
    return {"name": r[0], "mtype": r[1], "last_update": r[2]} if r else {}


def set_meta(code: str, **kw) -> None:
    m = get_meta(code)
    m.update({k: v for k, v in kw.items() if v is not None})
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO meta VALUES (?,?,?,?)",
                  (code, m.get("name"), m.get("mtype"), m.get("last_update")))


def last_update(codes: list[str]) -> str | None:
    vals = [get_meta(c).get("last_update") for c in codes]
    vals = [v for v in vals if v]
    return min(vals) if vals and len(vals) == len(codes) else None


# ================================================================ 規則模型回溯紀錄
REC_DAYS = 1000      # 規則分數保留最近 1000 個交易日（也是 ML 的特徵）


def update_rule_rec(code: str, px: pd.DataFrame, analyze_fn, days: int = REC_DAYS) -> int:
    """只計算還沒算過的日期；最後 2 天每次重算（盤中資料可能變動）。"""
    if px is None or len(px) < 140:
        return 0
    with conn() as c:
        have = {r[0] for r in c.execute("SELECT date FROM rule_rec WHERE code=?", (code,))}
    n = len(px)
    todo = [t for t in range(max(120, n - days), n) if _dstr(px.index[t]) not in have or t >= n - 2]
    rows = []
    for t in todo:
        r = analyze_fn(px.iloc[max(0, t + 1 - 250):t + 1])
        if not r:
            continue
        keep = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                for k, v in r.items() if k != "sub"}
        keep.update({f"sub_{k}": float(v) for k, v in r["sub"].items()})
        rows.append((code, _dstr(px.index[t]), json.dumps(keep, ensure_ascii=False)))
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO rule_rec VALUES (?,?,?)", rows)
    return len(rows)


def load_rule_rec(code: str, px: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
    """回傳與 evaluate.simulate 相同格式（含實際收盤與成敗）。"""
    import evaluate as EV
    with conn() as c:
        rows = c.execute("SELECT date, data FROM rule_rec WHERE code=? ORDER BY date", (code,)).fetchall()
    if not rows or px is None or px.empty:
        return pd.DataFrame()
    rec = pd.DataFrame([json.loads(d) for _, d in rows], index=pd.to_datetime([r[0] for r in rows]))
    rec.index.name = "date"
    pos = px.index.get_indexer(rec.index)
    rec = rec[pos >= 0]
    pos = pos[pos >= 0]
    if days:
        rec, pos = rec.iloc[-days:], pos[-days:]
    rec["pos"] = pos
    rec["close"] = px["Close"].values[pos]
    c_ = px["Close"].values
    n = len(px)
    for h in EV.HS:
        j = pos + h
        ok = j < n
        rec[f"act{h}"] = np.where(ok, c_[np.minimum(j, n - 1)], np.nan)
        rec[f"date{h}"] = pd.NaT
        rec.loc[ok, f"date{h}"] = px.index[j[ok]]
    return EV.add_results(rec)


# ================================================================ ML / 模型狀態 / 預測
def save_ml_oos(code: str, h: int, oos: pd.DataFrame) -> None:
    rows = [(code, _dstr(d), h, *(None if pd.isna(r[k]) else float(r[k]) for k in ("ridge", "gbdt", "ml")))
            for d, r in oos.iterrows()]
    with conn() as c:
        c.execute("DELETE FROM ml_oos WHERE code=? AND h=?", (code, h))
        c.executemany("INSERT OR REPLACE INTO ml_oos VALUES (?,?,?,?,?,?)", rows)


def load_ml_oos(code: str, h: int) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql_query("SELECT date, ridge, gbdt, ml FROM ml_oos WHERE code=? AND h=? ORDER BY date",
                               c, params=(code, h))
    if df.empty:
        return df
    df.index = pd.to_datetime(df.pop("date"))
    return df


def save_state(code: str, h: int, data: dict) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO model_state VALUES (?,?,?,?)",
                  (code, h, json.dumps(data, ensure_ascii=False, default=_json_default), _now()))


def load_state(code: str, h: int) -> dict:
    with conn() as c:
        r = c.execute("SELECT data, updated FROM model_state WHERE code=? AND h=?", (code, h)).fetchone()
    if not r:
        return {}
    d = json.loads(r[0])
    d["updated"] = r[1]
    return d


def save_forecast(code: str, model: str, date, data: dict) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO forecast VALUES (?,?,?,?,?)",
                  (code, model, _dstr(date), json.dumps(data, ensure_ascii=False, default=_json_default), _now()))


def load_forecasts(code: str) -> dict:
    with conn() as c:
        rows = c.execute("SELECT model, date, data, updated FROM forecast WHERE code=?", (code,)).fetchall()
    out = {}
    for m, d, js, up in rows:
        x = json.loads(js)
        x.update({"date": d, "updated": up})
        out[m] = x
    return out


def log_live(code: str, date, model: str, f: dict) -> None:
    """記錄當天預測；比資料庫最新日期還舊的不寫（過去不改寫）。"""
    d = _dstr(date)
    with conn() as c:
        last = c.execute("SELECT MAX(date) FROM live WHERE code=? AND model=?", (code, model)).fetchone()[0]
        if last and d < last:
            return
        c.execute("INSERT OR REPLACE INTO live VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (code, d, model, f["close"], f["atr"], f["p5"], f["lo5"], f["hi5"], f["p20"], f["lo20"], f["hi20"],
                   int(f["dir5"]), int(f["dir20"]), _now()))


def load_live(code: str, model: str, px: pd.DataFrame) -> pd.DataFrame:
    import evaluate as EV
    with conn() as c:
        rec = pd.read_sql_query("SELECT * FROM live WHERE code=? AND model=? ORDER BY date", c, params=(code, model))
    if rec.empty or px is None or px.empty:
        return pd.DataFrame()
    rec.index = pd.to_datetime(rec.pop("date"))
    pos = px.index.get_indexer(rec.index)
    rec = rec[pos >= 0]
    pos = pos[pos >= 0]
    if rec.empty:
        return pd.DataFrame()
    rec["pos"] = pos
    c_ = px["Close"].values
    n = len(px)
    for h in EV.HS:
        j = pos + h
        ok = j < n
        rec[f"act{h}"] = np.where(ok, c_[np.minimum(j, n - 1)], np.nan)
        rec[f"date{h}"] = pd.NaT
        rec.loc[ok, f"date{h}"] = px.index[j[ok]]
        rec[f"dirsig{h}"] = rec[f"dir{h}"]
    rec["score"] = 0.0
    rec["score20"] = 0.0
    rec["pattern"] = None
    rec["pattern_dir"] = 0
    return EV.add_results(rec.drop(columns=["dir5", "dir20"]))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def db_size() -> int:
    return DB.stat().st_size if DB.exists() else 0
