"""實際預測紀錄：每次打開看盤，就把每檔股票『當天』的預測存進 predictions.db（SQLite）。

- 同一天重複打開會覆寫當天那筆（以最後一次為準，通常是收盤後的完整資料）。
- 過去日期的紀錄永遠不會被修改，所以這是沒有事後偏誤的真實成績。
- 5 / 20 個交易日後，用實際收盤價自動對答案（evaluate.add_results）。
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as EV

DB = Path(__file__).parent / "predictions.db"
COLS = ["close", "atr", "score", "score20", "p5", "lo5", "hi5", "p20", "lo20", "hi20", "calibrated"]


def _conn():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS preds (
        code TEXT, date TEXT, close REAL, atr REAL, score REAL, score20 REAL,
        p5 REAL, lo5 REAL, hi5 REAL, p20 REAL, lo20 REAL, hi20 REAL,
        calibrated INTEGER, logged_at TEXT, PRIMARY KEY (code, date))""")
    return con


def log(code: str, date, r: dict | None, calibrated: bool = False) -> None:
    """記錄一筆預測。date = 預測所根據的最後一根 K 棒日期。"""
    if not r:
        return
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    try:
        with _conn() as con:
            last = con.execute("SELECT MAX(date) FROM preds WHERE code=?", (code,)).fetchone()[0]
            if last and d < last:          # 不改寫過去
                return
            con.execute(
                "INSERT OR REPLACE INTO preds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (code, d, r["close"], r["atr"], r["score"], r["score20"], float(r["p5"]), float(r["lo5"]),
                 float(r["hi5"]), float(r["p20"]), float(r["lo20"]), float(r["hi20"]), int(calibrated),
                 dt.datetime.now().isoformat(timespec="seconds")))
    except Exception:
        pass


def load(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """讀出某檔的實際紀錄，並用 df（日 K）對答案。"""
    try:
        with _conn() as con:
            rec = pd.read_sql_query("SELECT * FROM preds WHERE code=? ORDER BY date", con, params=(code,))
    except Exception:
        return pd.DataFrame()
    if rec.empty or df is None or df.empty:
        return pd.DataFrame()
    rec["date"] = pd.to_datetime(rec["date"])
    rec = rec.set_index("date")
    idx = df.index
    pos = idx.searchsorted(rec.index)
    rec["pos"] = pos
    c = df["Close"].values
    for h in EV.HS:
        j = pos + h
        ok = (pos < len(idx)) & (idx[np.minimum(pos, len(idx) - 1)] == rec.index) & (j < len(idx))
        rec[f"act{h}"] = np.where(ok, c[np.minimum(j, len(c) - 1)], np.nan)
        rec[f"date{h}"] = pd.NaT
        rec.loc[ok, f"date{h}"] = idx[j[ok]]
    rec["pattern"] = None
    rec["pattern_dir"] = 0
    return EV.add_results(rec)


def count() -> int:
    try:
        with _conn() as con:
            return con.execute("SELECT COUNT(*) FROM preds").fetchone()[0]
    except Exception:
        return 0
