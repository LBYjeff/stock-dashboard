"""趨勢預測引擎：依技術分析圖卡的規則，對日 K 打分並推估未來 5 日走向。

規則來源（使用者提供的 10 張圖卡）：
  1. K 棒型態：大/中/小 陽(陰)線、十字線、上下影線 → 多空力道、支撐/壓力
  2. 組合 K 線：晨星、錘頭線、吞噬、黃昏星、黑三兵
  3. 反轉型態：M頭、W底、頭肩頂/底、三重頂/底、倒V反轉、菱形頂
  4. 整理型態：上升/下降三角形、三角收斂、擴散三角形、箱型盤整、
              上升/下降楔形、上升/下跌旗形、上升/下降通道（突破方向決定買賣）
  5. 水平線：反彈兩次以上的價位畫橫線 → 突破壓力買、跌破支撐賣、支撐壓力轉換
  6. 移動平均線：黃金交叉買、死亡交叉賣；長期均線方向相反時「不要買/不要賣」

每個訊號帶方向(+多/-空/0中性)與權重，加總得綜合分數。
這是規則式評分，不是保證；回測函式會給出該股過去的實際命中率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as D

LOOKBACK = 250          # 分析最多用最近 250 根 K 棒
PIVOT_K = 3             # 轉折點：左右各 3 根內的最高/最低
HORIZON = 5             # 短期預測天數
HORIZON_LONG = 20       # 中期預測天數
SHORT_KINDS = ("K棒", "組合K線")   # 短線訊號：20 日預測只取 30% 權重
KINDS = ("K棒", "組合K線", "反轉型態", "整理型態", "水平線", "均線")
DEFAULT_W5 = {k: 1.0 for k in KINDS}
DRIFT = 0.25            # 每日預估移動 = 方向 × 強度 × DRIFT × ATR × 幅度係數
DEFAULT_W20 = {k: (0.3 if k in SHORT_KINDS else 1.0) for k in KINDS}


# ================================================================ 基礎工具
def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


def find_pivots(h: np.ndarray, l: np.ndarray, k: int = PIVOT_K):
    """回傳 (高點 index 清單, 低點 index 清單)。需要右側 k 根確認。"""
    ph, pl = [], []
    for i in range(k, len(h) - k):
        wh, wl = h[i - k:i + k + 1], l[i - k:i + k + 1]
        if np.argmax(wh) == k:
            ph.append(i)
        if np.argmin(wl) == k:
            pl.append(i)
    return ph, pl


def _sig(name, kind, d, w, note):
    return {"name": name, "kind": kind, "dir": int(np.sign(d)), "w": float(w) * (1 if d > 0 else -1 if d < 0 else 0),
            "note": note}


def _line(i0, y0, i1, y1, label, color, dash="solid"):
    return {"i0": int(i0), "y0": float(y0), "i1": int(i1), "y1": float(y1),
            "label": label, "color": color, "dash": dash}


RED, GREEN, GRAY, BLUE, ORANGE = "#e5484d", "#2f9e66", "#8b8d98", "#3e7bfa", "#f5a524"


# ================================================================ 1+2. K 棒與組合 K 線
def candle_signals(o, h, l, c, prior_chg):
    sigs = []
    body = np.abs(c - o)
    avg_body = np.mean(body[-21:-1]) or 1e-9
    rng = h - l

    def parts(i):
        b = abs(c[i] - o[i]); up = h[i] - max(o[i], c[i]); lo = min(o[i], c[i]) - l[i]
        return b, up, lo, max(rng[i], 1e-9)

    b, up, lo, r = parts(-1)
    bull = c[-1] > o[-1]

    # ---- 單根 K 棒
    if b <= 0.1 * r:
        d = -1 if prior_chg > 0.02 else 1 if prior_chg < -0.02 else 0
        sigs.append(_sig("十字線", "K棒", d, 0.5, "多空均衡，可能是市場轉折點"))
    elif b >= 1.5 * avg_body:
        sigs.append(_sig("大陽線" if bull else "大陰線", "K棒", 1 if bull else -1, 1,
                         "多方絕對優勢" if bull else "空方絕對優勢"))
    elif b >= 0.8 * avg_body:
        sigs.append(_sig("中陽線" if bull else "中陰線", "K棒", 1 if bull else -1, 0.5,
                         "多方力道" if bull else "空方力道"))

    hammer = lo >= 2 * b and up <= 0.35 * max(b, 0.1 * r) and lo >= 0.5 * r
    if hammer and prior_chg < -0.02:
        sigs.append(_sig("錘頭線", "組合K線", 1, 1.5, "下跌後出現長下影，下方支撐強（可靠度★★★）"))
    elif lo >= 2 * max(b, 0.1 * r) and lo >= 0.5 * r:
        sigs.append(_sig("長下影線", "K棒", 1, 0.5, "下方有支撐，影線越長越強"))
    if up >= 2 * max(b, 0.1 * r) and up >= 0.5 * r:
        sigs.append(_sig("長上影線", "K棒", -1, 0.5, "上方有壓力，壓力比下方支撐更強"))

    # ---- 兩根：吞噬
    b2 = abs(c[-2] - o[-2])
    if c[-2] < o[-2] and bull and o[-1] <= c[-2] and c[-1] >= o[-2] and b > b2 and prior_chg < 0:
        sigs.append(_sig("多頭吞噬", "組合K線", 1, 1.5, "陰陽燭吞噬：紅K完全包住前一根黑K（可靠度★★★）"))
    if c[-2] > o[-2] and not bull and o[-1] >= c[-2] and c[-1] <= o[-2] and b > b2 and prior_chg > 0:
        sigs.append(_sig("空頭吞噬", "組合K線", -1, 1.5, "黑K完全包住前一根紅K"))

    # ---- 三根：晨星 / 黃昏星 / 黑三兵
    b3, b2m = abs(c[-3] - o[-3]), abs(c[-2] - o[-2])
    mid3 = (o[-3] + c[-3]) / 2
    if c[-3] < o[-3] and b3 >= avg_body and b2m <= 0.5 * b3 and max(o[-2], c[-2]) <= c[-3] + 0.25 * b3 \
            and bull and c[-1] > mid3:
        sigs.append(_sig("晨星", "組合K線", 1, 2, "長黑、小K、長紅：底部反轉（可靠度★★★）"))
    if c[-3] > o[-3] and b3 >= avg_body and b2m <= 0.5 * b3 and min(o[-2], c[-2]) >= c[-3] - 0.25 * b3 \
            and not bull and c[-1] < mid3:
        sigs.append(_sig("黃昏星", "組合K線", -1, 2, "長紅、小K、長黑：頭部反轉（可靠度★★★）"))
    if all(c[i] < o[i] for i in (-3, -2, -1)) and c[-1] < c[-2] < c[-3] \
            and min(abs(c[i] - o[i]) for i in (-3, -2, -1)) >= 0.6 * avg_body:
        sigs.append(_sig("黑三兵", "組合K線", -1, 1.5, "連續三根長黑，空方持續"))
    return sigs


# ================================================================ 3. 反轉型態
def _reversal(hi, lo, close, piv, n, atr_v, top: bool):
    """偵測頭肩 / 三重 / 雙重。top=False 時傳入的是取負後的價格（偵測底部）。"""
    sgn = 1 if top else -1
    names = {"hs": ("頭肩頂", "頭肩底"), "tri": ("三重頂", "三重底"), "dbl": ("M頭（雙頂）", "W底（雙底）")}
    nm = lambda k: names[k][0 if top else 1]
    recent = [i for i in piv if i >= n - 100]
    c = close[-1]
    out = None

    def neck_between(a, b):
        j = a + int(np.argmin(lo[a:b + 1]))
        return j, lo[j]

    def result(key, peak, neck_now, pts, neck_pts, w_conf=3.0, w_form=1.5):
        if c > peak:                                   # 創新高(低) → 型態失效
            return None
        confirmed = c < neck_now
        d = -sgn
        height = peak - neck_now
        target = neck_now - height
        status = "已跌破頸線，型態確認" if top else "已突破頸線，型態確認"
        if not confirmed:
            status = "形成中，尚未跌破頸線" if top else "形成中，尚未突破頸線"
        name = nm(key)
        note = f"{status}；頸線 {sgn * neck_now:,.2f}"
        sig = _sig(name, "反轉型態", d, w_conf if confirmed else w_form, note)
        col = RED if top else GREEN
        lines = [_line(pts[k], sgn * hi[pts[k]], pts[k + 1], sgn * hi[pts[k + 1]], name, col, "dot")
                 for k in range(len(pts) - 1)]
        lines.append(_line(neck_pts[0], sgn * neck_pts[1], n - 1, sgn * neck_now, "頸線", col, "dash"))
        return {"sig": sig, "lines": lines, "target": sgn * target if confirmed else None}

    # 頭肩
    if len(recent) >= 3:
        a, b, d_ = recent[-3:]
        A, B, C = hi[a], hi[b], hi[d_]
        if d_ >= n - 35 and B - max(A, C) >= max(0.03 * abs(B), 1.5 * atr_v) and abs(A - C) <= 0.05 * abs(B):
            t1, v1 = neck_between(a, b)
            t2, v2 = neck_between(b, d_)
            slope = (v2 - v1) / max(t2 - t1, 1)
            neck_now = v2 + slope * (n - 1 - t2)
            out = result("hs", B, neck_now, [a, b, d_], (t1, v1))
            if out:
                return out
    # 三重
    if len(recent) >= 3:
        a, b, d_ = recent[-3:]
        pk = [hi[a], hi[b], hi[d_]]
        if d_ >= n - 35 and d_ - a >= 15 and (max(pk) - min(pk)) <= 0.03 * abs(np.mean(pk)):
            _, v1 = neck_between(a, b)
            t2, v2 = neck_between(b, d_)
            neck = min(v1, v2)
            depth = max(pk) - neck
            if depth >= max(0.04 * abs(max(pk)), 2 * atr_v):
                out = result("tri", max(pk), neck, [a, b, d_], (a, neck))
                if out:
                    return out
    # 雙重
    if len(recent) >= 2:
        a, b = recent[-2:]
        A, B = hi[a], hi[b]
        if b >= n - 35 and b - a >= 8 and abs(A - B) <= 0.03 * abs((A + B) / 2):
            t, v = neck_between(a, b)
            depth = max(A, B) - v
            if depth >= max(0.04 * abs(max(A, B)), 2 * atr_v):
                out = result("dbl", max(A, B), v, [a, b], (a, v), w_conf=3.0, w_form=1.5)
    return out


def inverse_v(h, l, c, atr_v):
    n = len(h)
    w = h[-12:]
    m = int(np.argmax(w)); mi = n - 12 + m
    if not (n - 10 <= mi <= n - 2):
        return None
    base_i = max(0, mi - 10) + int(np.argmin(l[max(0, mi - 10):mi + 1]))
    base = l[base_i]
    rise, drop = h[mi] - base, h[mi] - c[-1]
    if rise >= 0.08 * base and rise >= 3 * atr_v and drop >= 0.6 * rise:
        return {"sig": _sig("倒V型反轉", "反轉型態", -1, 2, f"急漲 {rise / base:.0%} 後快速回落 {drop / h[mi]:.0%}"),
                "lines": [_line(base_i, base, mi, h[mi], "倒V", RED, "dot"),
                          _line(mi, h[mi], n - 1, c[-1], "倒V", RED, "dot")]}
    return None


# ================================================================ 4. 整理型態（趨勢線）
def _fit(idx, vals):
    if len(idx) == 2:
        s = (vals[1] - vals[0]) / (idx[1] - idx[0])
        return s, vals[0] - s * idx[0]
    s, b = np.polyfit(idx, vals, 1)
    return s, b


def consolidation(h, l, c, ph, pl, atr_v, window=60):
    n = len(c)
    hs = [i for i in ph if i >= n - window][-4:]
    ls = [i for i in pl if i >= n - window][-4:]
    if len(hs) < 2 or len(ls) < 2 or max(hs[-1], ls[-1]) < n - 20:
        return None
    s = min(hs[0], ls[0])
    if n - 1 - s < 15:
        return None
    su, bu = _fit(hs, [h[i] for i in hs])
    sl, bl = _fit(ls, [l[i] for i in ls])
    u0, u1 = su * s + bu, su * (n - 1) + bu
    l0, l1 = sl * s + bl, sl * (n - 1) + bl
    if u0 <= l0:
        return None
    flat_tol = 1.2 * atr_v
    uf, lf = abs(u1 - u0) < flat_tol, abs(l1 - l0) < flat_tol
    uu, ld = u1 > u0, l1 < l0
    w0, w1 = u0 - l0, max(u1 - l1, 0)
    conv, div = w1 < 0.65 * w0, w1 > 1.35 * w0
    p0 = max(0, s - 15)
    pole = (c[s] - c[p0]) / c[p0]
    pole_up = pole >= 0.08 or c[s] - c[p0] >= 3 * atr_v
    pole_dn = pole <= -0.08 or c[p0] - c[s] >= 3 * atr_v

    if uf and lf:
        name, d, w = ("上升旗形", 1, 2) if pole_up else ("下跌旗形", -1, 2) if pole_dn else ("箱型盤整", 0, 0)
    elif uf and not ld:
        name, d, w = "上升三角形", 1, 2
    elif lf and not uu:
        name, d, w = "下降三角形", -1, 2
    elif not uu and not ld:
        name, d, w = "三角收斂", 0, 0
    elif uu and ld or uf or lf:
        name, d, w = "擴散三角形", 0, 0
    elif uu:   # 兩線皆上
        if conv:
            name, d, w = "上升楔形", -1, 2
        elif pole_up:
            name, d, w = "上升旗形", 1, 2
        elif pole_dn:
            name, d, w = "下跌旗形", -1, 2
        else:
            name, d, w = "上升通道", 0, 0
    else:      # 兩線皆下
        if conv:
            name, d, w = "下降楔形", 1, 2
        elif pole_up:
            name, d, w = "上升旗形", 1, 2
        elif pole_dn:
            name, d, w = "下跌旗形", -1, 2
        else:
            name, d, w = "下降通道", 0, 0
    if div and name not in ("擴散三角形",) and d == 0:
        name = "擴散三角形"

    brk = "up" if c[-1] > u1 + 0.2 * atr_v else "down" if c[-1] < l1 - 0.2 * atr_v else "in"
    if brk == "up":
        sig = _sig(f"{name}・向上突破", "整理型態", 1, (w + 1.5) if d >= 0 else 1.5,
                   "突破上緣，型態確認" if d >= 0 else "原屬空方型態卻向上突破，型態失敗轉多")
    elif brk == "down":
        sig = _sig(f"{name}・向下跌破", "整理型態", -1, (w + 1.5) if d <= 0 else 1.5,
                   "跌破下緣，型態確認" if d <= 0 else "原屬多方型態卻向下跌破，型態失敗轉空")
    elif d == 0:
        sig = _sig(name, "整理型態", 0, 0,
                   f"等待突破方向：上緣 {u1:,.2f}／下緣 {l1:,.2f}")
    else:
        sig = _sig(name, "整理型態", d, w * 0.6,
                   f"形成中：上緣 {u1:,.2f}／下緣 {l1:,.2f}，尚未突破")
    col = RED if sig["dir"] > 0 else GREEN if sig["dir"] < 0 else GRAY
    lines = [_line(s, u0, n - 1, u1, name + " 上緣", col, "dash"),
             _line(s, l0, n - 1, l1, name + " 下緣", col, "dash")]
    target = None
    if brk == "up" and sig["dir"] > 0:
        target = c[-1] + w0
    elif brk == "down" and sig["dir"] < 0:
        target = c[-1] - w0
    return {"sig": sig, "lines": lines, "target": target}


def diamond_top(h, l, c, ph, pl, atr_v, window=70):
    n = len(c)
    hs = [i for i in ph if i >= n - window]
    ls = [i for i in pl if i >= n - window]
    if len(hs) < 4 or len(ls) < 4:
        return None
    mid = n - window // 2
    h1, h2 = [i for i in hs if i < mid], [i for i in hs if i >= mid]
    l1, l2 = [i for i in ls if i < mid], [i for i in ls if i >= mid]
    if min(len(h1), len(h2), len(l1), len(l2)) < 2:
        return None
    s0 = n - window
    if c[s0] >= c[max(0, s0 - 20)] * 0.95:          # 需要先有漲勢
        pass
    else:
        return None
    a = _fit(h1, [h[i] for i in h1])[0]; b = _fit(l1, [l[i] for i in l1])[0]
    a2 = _fit(h2, [h[i] for i in h2])[0]; b2 = _fit(l2, [l[i] for i in l2])[0]
    if a > 0 and b < 0 and a2 < 0 and b2 > 0:
        s2, i2 = _fit(l2, [l[i] for i in l2])
        low_now = s2 * (n - 1) + i2
        if c[-1] < low_now - 0.2 * atr_v:
            return {"sig": _sig("菱形頂・跌破", "反轉型態", -1, 2.5, "先擴散後收斂的頭部型態，已跌破下緣"),
                    "lines": [_line(l2[0], l[l2[0]], n - 1, low_now, "菱形頂下緣", GREEN, "dash")]}
        return {"sig": _sig("菱形頂", "反轉型態", -1, 1, "先擴散後收斂，頭部可能成形，留意跌破"),
                "lines": [_line(l2[0], l[l2[0]], n - 1, low_now, "菱形頂下緣", GREEN, "dash")]}
    return None


# ================================================================ 5. 水平線支撐壓力
def horizontal_levels(h, l, c, o, ph, pl, atr_v, lookback=120):
    n = len(c)
    pts = sorted([h[i] for i in ph if i >= n - lookback] + [l[i] for i in pl if i >= n - lookback])
    if not pts:
        return [], [], None, None
    tol = max(0.012 * c[-1], 0.6 * atr_v)
    clusters, cur = [], [pts[0]]
    for p in pts[1:]:
        if p - np.mean(cur) <= tol:
            cur.append(p)
        else:
            clusters.append(cur); cur = [p]
    clusters.append(cur)
    levels = [(float(np.mean(cl)), len(cl)) for cl in clusters if len(cl) >= 2]   # 反彈兩次以上

    sigs = []
    prev = c[-6:-1]
    up_brk = [(L, k) for L, k in levels if c[-1] > L + 0.1 * atr_v and prev.min() < L]
    dn_brk = [(L, k) for L, k in levels if c[-1] < L - 0.1 * atr_v and prev.max() > L]
    if up_brk:
        L, k = max(up_brk, key=lambda x: (x[1], x[0]))
        sigs.append(_sig("突破水平壓力", "水平線", 1, 2, f"站上 {L:,.2f}（碰觸 {k} 次），壓力轉支撐"))
    if dn_brk:
        L, k = max(dn_brk, key=lambda x: (x[1], -x[0]))
        sigs.append(_sig("跌破水平支撐", "水平線", -1, 2, f"跌破 {L:,.2f}（碰觸 {k} 次），支撐轉壓力"))

    sup = max([x for x in levels if x[0] < c[-1]], default=None, key=lambda x: x[0])
    res = min([x for x in levels if x[0] > c[-1]], default=None, key=lambda x: x[0])
    if sup and not dn_brk and l[-3:].min() <= sup[0] + tol and c[-1] > o[-1]:
        sigs.append(_sig("支撐反彈", "水平線", 1, 1, f"回測支撐 {sup[0]:,.2f} 後收紅"))
    if res and not up_brk and h[-3:].max() >= res[0] - tol and c[-1] < o[-1]:
        sigs.append(_sig("壓力回落", "水平線", -1, 1, f"碰壓力 {res[0]:,.2f} 後收黑"))
    return sigs, levels, sup, res


# ================================================================ 6. 移動平均線
def ma_signals(c):
    s = pd.Series(c)
    ma = {k: s.rolling(k).mean().values for k in (5, 20, 60)}
    sigs = []

    def cross(a, b, look=3):
        for j in range(1, look + 1):
            if np.isnan(b[-j - 1]):
                return None, 0
            if a[-j] > b[-j] and a[-j - 1] <= b[-j - 1]:
                return "golden", j
            if a[-j] < b[-j] and a[-j - 1] >= b[-j - 1]:
                return "death", j
        return None, 0

    for short, long_, w in ((5, 20, 2), (20, 60, 2)):
        kind, j = cross(ma[short], ma[long_])
        slope = ma[long_][-1] - ma[long_][-6]
        tag = f"MA{short}/MA{long_}"
        if kind == "golden":
            if slope > 0:
                sigs.append(_sig(f"黃金交叉 {tag}", "均線", 1, w, f"{j} 日前短均線向上穿越長均線，長均線上升"))
            else:
                sigs.append(_sig(f"黃金交叉 {tag}（不要買）", "均線", 0, 0, "長期均線仍在下跌，交叉可信度低"))
        elif kind == "death":
            if slope < 0:
                sigs.append(_sig(f"死亡交叉 {tag}", "均線", -1, w, f"{j} 日前短均線向下跌破長均線，長均線下彎"))
            else:
                sigs.append(_sig(f"死亡交叉 {tag}（不要賣）", "均線", 0, 0, "長期基準仍上升中，交叉可信度低"))

    m5, m20, m60 = ma[5][-1], ma[20][-1], ma[60][-1]
    up20 = ma[20][-1] > ma[20][-6]
    if m5 > m20 > m60 and up20:
        sigs.append(_sig("均線多頭排列", "均線", 1, 1, "MA5 > MA20 > MA60，且 MA20 上揚"))
    elif m5 < m20 < m60 and not up20:
        sigs.append(_sig("均線空頭排列", "均線", -1, 1, "MA5 < MA20 < MA60，且 MA20 下彎"))
    return sigs


# ================================================================ 主程式
def label_of(score):
    if score >= 5:
        return "強烈偏多"
    if score >= 2:
        return "偏多"
    if score <= -5:
        return "強烈偏空"
    if score <= -2:
        return "偏空"
    return "盤整觀望"


def analyze(df: pd.DataFrame, light: bool = False, w5: dict | None = None,
            w20: dict | None = None) -> dict | None:
    """df：日 K（Open/High/Low/Close/Volume）。
    light=True 時只回傳分數與預估價（回測用，較快）。
    w5 / w20：各訊號類別的權重倍數（個股校正用），None = 預設。"""
    if df is None or len(df) < 80:
        return None
    df = df.iloc[-LOOKBACK:]
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    n = len(c)
    atr_v = float(atr(df)[-1]) or 1e-9
    ph, pl = find_pivots(h, l)
    prior_chg = c[-2] / c[-7] - 1 if n > 7 else 0

    sigs, lines, targets = [], [], []

    def take(res):
        if res:
            sigs.append(res["sig"]); lines.extend(res.get("lines", []))
            if res.get("target") is not None:
                targets.append((res["sig"]["name"], float(res["target"])))

    sigs += candle_signals(o, h, l, c, prior_chg)
    take(_reversal(h, l, c, ph, n, atr_v, top=True))
    take(_reversal(-l, -h, -c, pl, n, atr_v, top=False))
    take(inverse_v(h, l, c, atr_v))
    take(diamond_top(h, l, c, ph, pl, atr_v))
    take(consolidation(h, l, c, ph, pl, atr_v))
    hs, levels, sup, res = horizontal_levels(h, l, c, o, ph, pl, atr_v)
    sigs += hs
    sigs += ma_signals(c)

    sub = {k: round(sum(x["w"] for x in sigs if x["kind"] == k), 3) for k in KINDS}
    m5 = dict(DEFAULT_W5); m5.update(w5 or {})
    m20 = dict(DEFAULT_W20); m20.update(w20 or {})
    score = round(sum(sub[k] * m5[k] for k in KINDS), 2)
    score20 = round(sum(sub[k] * m20[k] for k in KINDS), 2)
    pj = project(float(c[-1]), atr_v, score, score20, targets, m5.get("幅度", 1.0), m20.get("幅度", 1.0))

    out = {"score": score, "label": label_of(score), "score20": score20, "label20": label_of(score20),
           "close": float(c[-1]), "atr": atr_v, "sub": sub,
           "p5": pj["mid5"][-1], "lo5": pj["lo5"][-1], "hi5": pj["hi5"][-1],
           "p20": pj["mid20"][-1], "lo20": pj["lo20"][-1], "hi20": pj["hi20"][-1]}
    if light:
        pat = [x for x in sigs if x["kind"] in ("反轉型態", "整理型態") and abs(x["w"]) >= 1.5]
        top = max(pat, key=lambda x: abs(x["w"])) if pat else None
        out["pattern"] = top["name"] if top else None
        out["pattern_dir"] = top["dir"] if top else 0
        return out

    fut = D.next_sessions(df.attrs.get("cal"), df.index[-1], HORIZON_LONG)   # 只排交易日，跳過休市日
    shown = [dict(x, w=round(x["w"] * m5[x["kind"]], 2),
                  note=x["note"] + (f"（個股校正 ×{m5[x['kind']]:g}）" if m5[x["kind"]] != 1 else ""))
             for x in sigs]
    out.update({
        "strength": pj["strength5"],
        "strength20": pj["strength20"],
        "signals": sorted(shown, key=lambda x: -abs(x["w"])),
        "lines": lines,
        "levels": levels,
        "support": sup[0] if sup else None,
        "resistance": res[0] if res else None,
        "targets": targets,
        "index": df.index,
        "weights5": m5, "weights20": m20,
        "proj": pd.DataFrame({"mid": pj["mid5"], "hi": pj["hi5"], "lo": pj["lo5"]}, index=fut[:HORIZON]),
        "proj20": pd.DataFrame({"mid": pj["mid20"], "hi": pj["hi20"], "lo": pj["lo20"]}, index=fut),
    })
    return out


def project(close: float, atr_v: float, score: float, score20: float, targets=(),
            scale5: float = 1.0, scale20: float = 1.0) -> dict:
    """5 日：每日 = 方向×強度×DRIFT×ATR×幅度係數；20 日：影響逐日 ×0.92 遞減，
    有同向型態目標價時往目標靠攏一半（同樣乘幅度係數）。區間 = ±0.9 ATR×√天數。"""
    t = np.arange(1, HORIZON + 1)
    st5 = min(abs(score) / 8, 1.0)
    d5 = np.sign(score) * st5 * DRIFT * atr_v * scale5 if abs(score) >= 2 else 0.0
    mid5 = close + d5 * t
    band5 = 0.9 * atr_v * np.sqrt(t)

    t20 = np.arange(1, HORIZON_LONG + 1)
    st20 = min(abs(score20) / 8, 1.0)
    d20 = np.sign(score20) * st20 * DRIFT * atr_v * scale20 if abs(score20) >= 2 else 0.0
    mid20 = close + d20 * np.cumsum(0.92 ** (t20 - 1))
    same_dir = [v for _, v in targets if np.sign(v - close) == np.sign(score20) and abs(score20) >= 2]
    if same_dir:
        gap = same_dir[0] - mid20[-1]
        if np.sign(gap) == np.sign(score20):
            mid20 = mid20 + gap * 0.5 * scale20 * t20 / HORIZON_LONG
    band20 = 0.9 * atr_v * np.sqrt(t20)
    return {"mid5": mid5, "lo5": mid5 - band5, "hi5": mid5 + band5, "strength5": round(st5 * 100),
            "mid20": mid20, "lo20": mid20 - band20, "hi20": mid20 + band20, "strength20": round(st20 * 100)}


def _hit_stats(sc, fw):
    bull, bear = sc >= 2, sc <= -2
    return {
        "days": int(len(fw)),
        "base_up": float((fw > 0).mean()),
        "bull_n": int(bull.sum()), "bull_hit": float((fw[bull] > 0).mean()) if bull.any() else None,
        "bull_ret": float(fw[bull].mean()) if bull.any() else None,
        "bear_n": int(bear.sum()), "bear_hit": float((fw[bear] < 0).mean()) if bear.any() else None,
        "bear_ret": float(fw[bear].mean()) if bear.any() else None,
    }


def backtest(df: pd.DataFrame, days: int = 250) -> dict | None:
    """用過去每一天『當時可見的資料』跑 analyze，檢查 5 日 / 20 日後實際漲跌。
    回傳 {5: 統計, 20: 統計}。"""
    if df is None or len(df) < 140:
        return None
    c = df["Close"].values
    n = len(c)
    rows = []
    for t in range(max(120, n - days - HORIZON), n - HORIZON):
        r = analyze(df.iloc[max(0, t + 1 - LOOKBACK):t + 1], light=True)
        if r:
            f20 = c[t + HORIZON_LONG] / c[t] - 1 if t + HORIZON_LONG < n else np.nan
            rows.append((r["score"], r["score20"], c[t + HORIZON] / c[t] - 1, f20))
    if not rows:
        return None
    a = np.array(rows, dtype=float)
    out = {5: _hit_stats(a[:, 0], a[:, 2])}
    m = ~np.isnan(a[:, 3])
    out[20] = _hit_stats(a[m, 1], a[m, 3]) if m.sum() >= 20 else None
    return out
