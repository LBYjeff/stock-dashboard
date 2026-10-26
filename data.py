"""資料層：美股/台股報價與 K 線 (yfinance)，台股三大法人、融資券 (FinMind)。

所有函式都回傳 pandas DataFrame / dict，抓不到資料時回傳空 DataFrame 或 None，
不丟例外，讓畫面可以優雅地顯示「無資料」。
"""
from __future__ import annotations

import datetime as dt
import re
from functools import lru_cache

import pandas as pd
import requests
import yfinance as yf

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TIMEOUT = 20

try:
    import exchange_calendars as xcals
except Exception:          # 沒裝時退回「週一到週五」
    xcals = None


# ---------------------------------------------------------------- 交易日曆
def cal_name(code: str) -> str:
    """台股(含加權指數) → XTAI 臺灣證交所；其餘 → XNYS 紐約證交所。"""
    c = code.strip().upper()
    return "XTAI" if is_tw(c) or c in ("^TWII", "^TWOII") else "XNYS"


@lru_cache(maxsize=4)
def _cal(name: str):
    if xcals is None:
        return None
    try:
        return xcals.get_calendar(name)
    except Exception:
        return None


def clean_sessions(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """剔除休市日的資料：
    1) 不在交易所日曆上的日期（國定假日、春節、週末等）
    2) 成交量 0 且最高 = 最低（颱風假等臨時休市，資料源補上的空白列）"""
    if df is None or df.empty:
        return df
    name = cal_name(code)
    keep = pd.Series(True, index=df.index)
    cal = _cal(name)
    if cal is not None:
        first, last = cal.first_session, cal.last_session
        sess = cal.sessions
        inside = (df.index >= first) & (df.index <= last)
        keep &= ~inside | df.index.isin(sess)
    keep &= ~((df["Volume"].fillna(0) <= 0) & (df["High"] == df["Low"]))
    out = df[keep.values].copy()
    out.attrs["cal"] = name
    return out


def next_sessions(name: str | None, last_date, n: int) -> pd.DatetimeIndex:
    """last_date 之後的 n 個交易日（跳過休市日）。"""
    start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    cal = _cal(name) if name else None
    days: list = []
    if cal is not None:
        try:
            end = min(start + pd.Timedelta(days=n * 2 + 40), cal.last_session)
            if end >= start:
                days = list(cal.sessions_in_range(start, end)[:n])
        except Exception:
            days = []
    if len(days) < n:      # 超出日曆範圍才用週一到週五補
        nxt = (days[-1] + pd.Timedelta(days=1)) if days else start
        days += list(pd.bdate_range(nxt, periods=n - len(days)))
    return pd.DatetimeIndex(days)


# ---------------------------------------------------------------- 代號判斷
def is_tw(code: str) -> bool:
    """純數字(可帶英文字尾，如 00679B、2881A)視為台股，其餘視為美股。"""
    return bool(re.fullmatch(r"\d{4,6}[A-Z]?", code.strip().upper()))


def normalize(code: str) -> str:
    code = code.strip().upper()
    for suf in (".TW", ".TWO"):
        if code.endswith(suf):
            code = code[: -len(suf)]
    return code


# ---------------------------------------------------------------- FinMind
def finmind(dataset: str, data_id: str | None = None, start: str | None = None,
            end: str | None = None, token: str = "") -> pd.DataFrame:
    params = {"dataset": dataset}
    if data_id:
        params["data_id"] = data_id
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        params["token"] = token
    try:
        r = requests.get(FINMIND_URL, params=params, headers=headers, timeout=TIMEOUT)
        js = r.json()
    except Exception:
        return pd.DataFrame()
    if js.get("status") != 200 or not js.get("data"):
        return pd.DataFrame()
    return pd.DataFrame(js["data"])


def tw_stock_info(token: str = "") -> pd.DataFrame:
    """台股代號表：stock_id, stock_name, type(twse/tpex)。"""
    df = finmind("TaiwanStockInfo", token=token)
    if df.empty:
        return df
    return df.drop_duplicates("stock_id")[["stock_id", "stock_name", "type"]]


# ---------------------------------------------------------------- 價格
def _clean_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    return df[~df.index.duplicated(keep="last")].sort_index()


def _finmind_price(code: str, start: str, token: str) -> pd.DataFrame:
    df = finmind("TaiwanStockPrice", code, start=start, token=token)
    if df.empty:
        return df
    df = df.rename(columns={"open": "Open", "max": "High", "min": "Low",
                            "close": "Close", "Trading_Volume": "Volume"})
    df.index = pd.to_datetime(df["date"])
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df[df["Close"] > 0].sort_index()


def yahoo_symbol(code: str, market_type: str | None = None) -> str:
    if not is_tw(code):
        return code
    return f"{code}.TWO" if market_type == "tpex" else f"{code}.TW"


def price_history(code: str, period: str = "2y", market_type: str | None = None,
                  token: str = "") -> pd.DataFrame:
    """日 K：Open/High/Low/Close/Volume。台股 Volume 為「股」。"""
    sym = yahoo_symbol(code, market_type)
    df = pd.DataFrame()
    try:
        df = _clean_yf(yf.Ticker(sym).history(period=period, auto_adjust=False))
    except Exception:
        pass
    if df.empty and is_tw(code):
        # 上市/上櫃判斷錯誤時換個後綴再試
        alt = f"{code}.TWO" if sym.endswith(".TW") else f"{code}.TW"
        try:
            df = _clean_yf(yf.Ticker(alt).history(period=period, auto_adjust=False))
        except Exception:
            pass
    if df.empty and is_tw(code):
        years = {"1y": 1, "2y": 2, "5y": 5, "10y": 10, "max": 30}.get(period, 2)
        start = (dt.date.today() - dt.timedelta(days=365 * years + 10)).isoformat()
        df = _finmind_price(code, start, token)
    return clean_sessions(df, code)


def us_name(code: str) -> str:
    try:
        info = yf.Ticker(code).info
        return info.get("shortName") or info.get("longName") or code
    except Exception:
        return code


# ---------------------------------------------------------------- 三大法人
INST_GROUP = {
    "Foreign_Investor": "外資",
    "Foreign_Dealer_Self": "外資",
    "Investment_Trust": "投信",
    "Dealer_self": "自營商",
    "Dealer_Hedging": "自營商",
}


def institutional(code: str, days: int = 90, token: str = "") -> pd.DataFrame:
    """每日買賣超(張)：index=date, columns=外資/投信/自營商/合計。"""
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    df = finmind("TaiwanStockInstitutionalInvestorsBuySell", code, start=start, token=token)
    if df.empty:
        return df
    df["group"] = df["name"].map(INST_GROUP)
    df = df.dropna(subset=["group"])
    df["net"] = (df["buy"].astype(float) - df["sell"].astype(float)) / 1000
    out = df.pivot_table(index="date", columns="group", values="net", aggfunc="sum").fillna(0)
    for c in ("外資", "投信", "自營商"):
        if c not in out:
            out[c] = 0.0
    out = out[["外資", "投信", "自營商"]].round(0)
    out["合計"] = out.sum(axis=1)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def streak(series: pd.Series) -> int:
    """連續買超(+)或賣超(-)天數。"""
    s = series.dropna()
    if s.empty or s.iloc[-1] == 0:
        return 0
    sign = 1 if s.iloc[-1] > 0 else -1
    n = 0
    for v in reversed(s.tolist()):
        if v * sign > 0:
            n += 1
        else:
            break
    return n * sign


# ---------------------------------------------------------------- 融資融券
def margin(code: str, days: int = 120, token: str = "") -> pd.DataFrame:
    """融資/融券餘額(張)、增減、券資比、融資使用率。"""
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    df = finmind("TaiwanStockMarginPurchaseShortSale", code, start=start, token=token)
    if df.empty:
        return df
    df.index = pd.to_datetime(df["date"])
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["融資餘額"] = num("MarginPurchaseTodayBalance")
    out["融資增減"] = out["融資餘額"] - num("MarginPurchaseYesterdayBalance")
    out["融資限額"] = num("MarginPurchaseLimit")
    out["融券餘額"] = num("ShortSaleTodayBalance")
    out["融券增減"] = out["融券餘額"] - num("ShortSaleYesterdayBalance")
    out["券資比%"] = (out["融券餘額"] / out["融資餘額"].replace(0, pd.NA) * 100).astype(float).round(2)
    out["融資使用率%"] = (out["融資餘額"] / out["融資限額"].replace(0, pd.NA) * 100).astype(float).round(2)
    return out.sort_index()


# ---------------------------------------------------------------- 指標
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (5, 20, 60):
        ma = df["Close"].rolling(n).mean()
        df[f"MA{n}"] = ma
        df[f"BIAS{n}"] = (df["Close"] - ma) / ma * 100
    df["VOL_MA5"] = df["Volume"].rolling(5).mean()
    return df


def high_low_stats(full: pd.DataFrame) -> dict | None:
    """歷史 / 52 週 高低點與距離。full 需為完整歷史(period=max)。"""
    if full is None or full.empty:
        return None
    close = float(full["Close"].iloc[-1])
    last = full.index[-1]
    yr = full[full.index > last - pd.Timedelta(days=365)]

    def pack(d):
        hi_d, lo_d = d["High"].idxmax(), d["Low"].idxmin()
        hi, lo = float(d["High"].max()), float(d["Low"].min())
        return {"high": hi, "high_date": hi_d, "low": lo, "low_date": lo_d,
                "from_high": (close / hi - 1) * 100, "from_low": (close / lo - 1) * 100}

    return {"close": close, "first_date": full.index[0], "all": pack(full), "52w": pack(yr)}


def quote_row(code: str, df: pd.DataFrame) -> dict:
    """自選股表格的一列。"""
    if df is None or len(df) < 2:
        return {"代號": code, "收盤": None}
    df = add_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    chg = last["Close"] - prev["Close"]
    tw = is_tw(code)
    vol = last["Volume"] / 1000 if tw else last["Volume"]
    yr = df[df.index > df.index[-1] - pd.Timedelta(days=365)]
    hi52 = yr["High"].max()
    vol_ma = df["Volume"].iloc[-6:-1].mean()
    return {
        "代號": code,
        "日期": df.index[-1].strftime("%m/%d"),
        "收盤": round(float(last["Close"]), 2),
        "漲跌": round(float(chg), 2),
        "漲跌%": round(float(chg / prev["Close"] * 100), 2),
        "成交量": float(vol),
        "量比": round(float(last["Volume"] / vol_ma), 2) if vol_ma else None,
        "MA20乖離%": round(float(last["BIAS20"]), 2) if pd.notna(last["BIAS20"]) else None,
        "距52週高%": round(float((last["Close"] / hi52 - 1) * 100), 2),
    }


# ---------------------------------------------------------------- 預測檢討用
def market_symbol(code: str) -> str:
    """個股對應的大盤指數：台股 → 加權指數，美股 → S&P 500。"""
    return "^TWII" if is_tw(code) else "^GSPC"


def dividends(code: str, market_type: str | None = None) -> pd.Series:
    """除權息紀錄（每股現金股利，日期為除息日）。抓不到回傳空 Series。"""
    syms = [yahoo_symbol(code, market_type)]
    if is_tw(code):
        syms.append(f"{code}.TWO" if syms[0].endswith(".TW") else f"{code}.TW")
    for sym in syms:
        try:
            s = yf.Ticker(sym).dividends
        except Exception:
            continue
        if s is not None and len(s):
            idx = pd.to_datetime(s.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            return pd.Series(s.values.astype(float), index=idx.normalize()).sort_index()
    return pd.Series(dtype=float)
