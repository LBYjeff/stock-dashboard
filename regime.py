"""大盤環境（多空格局）與高波動判斷。每一天只用「當天收盤以前」的資料，可直接用於回溯驗證。

格局分數 regime（-1 空頭 ～ +1 多頭）由五項組成並扣加背離：
  1. 指數在半年線（MA120）之上 / 之下
  2. 半年線方向（20 日斜率）
  3. 月線 RSI(6)：當月尚未結束時，用「今天收盤」當本月收盤（不偷看月底）
  4. 週線 RSI(6)
  5. 指數在年線（MA240）之上 / 之下
  背離：月線價格創高但 RSI 沒創高（空頭背離）→ 每次 -0.25；反之多頭背離 +0.25（近 12 個月內）

用途：
  - 規則／校正模型：訊號方向與大盤格局相反時降權（短反彈難 V 轉 → 逆勢的多方訊號打折）
  - 機器學習：當作特徵，讓每檔股票自己學會對大盤的敏感度
  - 高波動（波動度暴增、指數單日大振幅、VIX 高）→ 預測改為「觀望」
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAMP5, DAMP20 = 0.0, 1.0          # 逆勢訊號打折強度：回測顯示只對 20 日有幫助（5 日會降低方向命中率）


def _rsi_from_closes(c: np.ndarray, n: int) -> float:
    if len(c) < n + 1:
        return np.nan
    d = np.diff(c[-(n + 1):])
    up, dn = d.clip(min=0).mean(), (-d).clip(min=0).mean()
    return 100.0 if dn == 0 and up > 0 else 50.0 if up == dn == 0 else 100 * up / (up + dn)


def _period_rsi(close: pd.Series, freq: str, n: int = 6) -> pd.Series:
    """週線/月線 RSI，逐日計算：已結束的週期用該期收盤，進行中的週期用今天收盤。"""
    per = close.index.to_period(freq)
    vals = close.values
    out = np.full(len(close), np.nan)
    done: list[float] = []          # 已結束週期的收盤
    cur = None
    for i in range(len(close)):
        if cur is not None and per[i] != cur:
            done.append(vals[i - 1])
        cur = per[i]
        out[i] = _rsi_from_closes(np.array(done[-n:] + [vals[i]]), n)
    return pd.Series(out, index=close.index)


def _monthly_divergences(close: pd.Series, n: int = 6, look: int = 12):
    """近 look 個月內確認的月線空頭/多頭背離次數（逐日，只用已結束的月份）。"""
    m = close.groupby(close.index.to_period("M")).last()
    mc = m.values
    rs = np.array([_rsi_from_closes(mc[:i + 1], n) for i in range(len(mc))])
    bear_at, bull_at = np.zeros(len(mc)), np.zeros(len(mc))   # 以「確認月份」記錄
    last_hi = last_lo = None
    for i in range(1, len(mc) - 1):
        if np.isnan(rs[i]):
            continue
        if mc[i] >= mc[i - 1] and mc[i] > mc[i + 1]:            # 月線高點（下個月確認）
            if last_hi is not None and i - last_hi <= 24 and mc[i] > mc[last_hi] and rs[i] < rs[last_hi] - 3:
                bear_at[i + 1] += 1
            last_hi = i
        if mc[i] <= mc[i - 1] and mc[i] < mc[i + 1]:            # 月線低點
            if last_lo is not None and i - last_lo <= 24 and mc[i] < mc[last_lo] and rs[i] > rs[last_lo] + 3:
                bull_at[i + 1] += 1
            last_lo = i
    # 第 k 個月內可見的 = 確認月份 < 本月（本月剛確認的要等月底，保守起見只算前一個月以前）
    bear_cum = pd.Series(bear_at, index=m.index).rolling(look, min_periods=1).sum().shift(1).fillna(0)
    bull_cum = pd.Series(bull_at, index=m.index).rolling(look, min_periods=1).sum().shift(1).fillna(0)
    per = close.index.to_period("M")
    return (pd.Series(bear_cum.reindex(per).values, index=close.index),
            pd.Series(bull_cum.reindex(per).values, index=close.index))


def market_regime(mkt: pd.DataFrame, vix: pd.DataFrame | None = None) -> pd.DataFrame:
    """mkt：指數日 K。回傳逐日 regime、各組成與波動指標。"""
    if mkt is None or len(mkt) < 260:
        return pd.DataFrame()
    c = mkt["Close"].astype(float)
    ma120, ma240 = c.rolling(120).mean(), c.rolling(240).mean()
    f = pd.DataFrame(index=mkt.index)
    f["above120"] = np.where(c > ma120, 1.0, -1.0)
    f["slope120"] = np.sign(ma120 - ma120.shift(20))
    f["mrsi6"] = _period_rsi(c, "M")
    f["wrsi6"] = _period_rsi(c, "W-FRI")
    f["above240"] = np.where(c > ma240, 1.0, -1.0)
    f["bear_div"], f["bull_div"] = _monthly_divergences(c)
    comp = pd.concat([f["above120"], f["slope120"], ((f["mrsi6"] - 50) / 50).clip(-1, 1),
                      ((f["wrsi6"] - 50) / 50).clip(-1, 1), f["above240"]], axis=1)
    raw = comp.mean(axis=1) - 0.25 * f["bear_div"].clip(upper=2) + 0.25 * f["bull_div"].clip(upper=2)
    f["regime"] = raw.clip(-1, 1)
    f.loc[ma240.isna(), "regime"] = np.nan

    ret = c.pct_change()
    rv = ret.rolling(20).std() * np.sqrt(252)
    f["rv20"] = rv
    f["vol_ratio"] = rv / rv.rolling(250, min_periods=120).median()
    f["range_pct"] = (mkt["High"] - mkt["Low"]) / c.shift()
    f["vix"] = np.nan
    if vix is not None and not vix.empty:
        f["vix"] = vix["Close"].reindex(mkt.index, method="ffill").values
    f["high_vol"] = ((f["vol_ratio"] > 1.8) | (f["range_pct"] > 0.03) | (f["vix"] > 30)).astype(float)
    return f


def damp(score, r, k):
    """訊號方向與大盤格局相反時打折：score × (1 − k·|r|)。r 為 NaN 時不調整。"""
    score = np.asarray(score, float)
    r = np.nan_to_num(np.asarray(r, float), nan=0.0)
    oppose = np.sign(score) * np.sign(r) < 0
    return np.where(oppose, score * (1 - k * np.abs(r)), score)


def label(r) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "資料不足"
    return "多頭" if r >= 0.5 else "偏多" if r >= 0.15 else "空頭" if r <= -0.5 else "偏空" if r <= -0.15 else "中性"


def summary(reg: pd.DataFrame) -> dict | None:
    """給畫面顯示的大盤環境摘要（最後一天）。"""
    if reg is None or reg.empty:
        return None
    x = reg.iloc[-1]
    notes = []
    if x["bear_div"] > 0:
        notes.append(f"近一年月線空頭背離 {int(x['bear_div'])} 次")
    if x["bull_div"] > 0:
        notes.append(f"近一年月線多頭背離 {int(x['bull_div'])} 次")
    notes.append("指數在半年線之上" if x["above120"] > 0 else "指數在半年線之下")
    return {"date": f"{reg.index[-1]:%Y-%m-%d}", "regime": None if np.isnan(x["regime"]) else float(x["regime"]),
            "label": label(x["regime"]), "mrsi6": float(x["mrsi6"]), "wrsi6": float(x["wrsi6"]),
            "vol_ratio": None if np.isnan(x["vol_ratio"]) else float(x["vol_ratio"]),
            "range_pct": None if np.isnan(x["range_pct"]) else float(x["range_pct"]),
            "vix": None if np.isnan(x["vix"]) else float(x["vix"]),
            "high_vol": bool(x["high_vol"]), "notes": notes}


def chip_summary(mac: pd.DataFrame) -> list[dict] | None:
    """大盤籌碼摘要（給畫面）：外資現貨、外資台指期淨單、大盤融資、台幣匯率。"""
    if mac is None or mac.empty:
        return None
    out = []

    def last(k):
        s = mac[k].dropna() if k in mac else pd.Series(dtype=float)
        return s

    s = last("tw_foreign_net")
    if len(s) >= 5:
        out.append({"name": "外資現貨買賣超", "value": f"{s.iloc[-1]:+,.0f} 億", "sub": f"5 日累計 {s.iloc[-5:].sum():+,.0f} 億",
                    "dir": int(np.sign(s.iloc[-5:].sum())), "date": f"{s.index[-1]:%m/%d}"})
    s = last("tx_foreign_oi")
    if len(s) >= 6:
        out.append({"name": "外資台指期淨未平倉", "value": f"{s.iloc[-1]:+,.0f} 口", "sub": f"5 日變化 {s.iloc[-1] - s.iloc[-6]:+,.0f} 口",
                    "dir": int(np.sign(s.iloc[-1])), "date": f"{s.index[-1]:%m/%d}"})
    s = last("tw_margin_money")
    if len(s) >= 6:
        ch = s.iloc[-1] / s.iloc[-6] - 1
        out.append({"name": "大盤融資餘額", "value": f"{s.iloc[-1]:,.0f} 億", "sub": f"5 日 {ch:+.1%}"
                    + ("（融資減肥中）" if ch < -0.03 else "（融資增加）" if ch > 0.03 else ""),
                    "dir": 0, "date": f"{s.index[-1]:%m/%d}"})
    s = last("usdtwd")
    if len(s) >= 6:
        ch = s.iloc[-1] / s.iloc[-6] - 1
        out.append({"name": "美元兌台幣", "value": f"{s.iloc[-1]:.3f}", "sub": f"5 日 {ch:+.2%}・" + ("台幣貶值" if ch > 0 else "台幣升值"),
                    "dir": -int(np.sign(ch)), "date": f"{s.index[-1]:%m/%d}"})
    return out or None
