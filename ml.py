"""第二階段：每檔股票自己的機器學習模型（純 numpy，不需額外安裝套件）。

目標：預測 N 日後漲跌幅（以 ATR 為單位，讓高低波動股票可比較）。
模型：脊迴歸（線性、穩定）＋ 梯度提升決策樹（可學非線性條件），兩者平均 = 「ML 模型」。
驗證：前推驗證（walk-forward）——每 40 個交易日只用『當時已知答案』的資料重新訓練，
      再預測接下來 40 天，得到完全沒看過答案的樣本外預測，用來和規則模型公平比較。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import trend as T

FEATURE_NAMES = {
    "ret1": "前 1 日漲跌", "ret5": "近 5 日漲跌", "ret20": "近 20 日漲跌", "ret60": "近 60 日漲跌",
    "bias5": "MA5 乖離", "bias20": "MA20 乖離", "bias60": "MA60 乖離",
    "atr_pct": "波動度（ATR%）", "vol_ratio": "量比（今日/20日均量）", "vol5_20": "5日/20日均量",
    "rsi14": "RSI(14)", "pos20": "20 日區間位置", "dist_hi250": "距 52 週高", "dist_lo250": "距 52 週低",
    "ma20_slope": "MA20 斜率", "ma60_slope": "MA60 斜率", "gap": "跳空幅度", "body": "K 棒實體比例",
    "upper": "上影線比例", "lower": "下影線比例",
    "mkt_ret5": "大盤近 5 日", "mkt_ret20": "大盤近 20 日", "mkt_bias20": "大盤 MA20 乖離",
    "rel20": "相對大盤強弱（20日）",
    "inst5": "法人 5 日買賣超/均量", "inst20": "法人 20 日買賣超/均量", "foreign5": "外資 5 日/均量",
    "trust5": "投信 5 日/均量", "margin_chg5": "融資 5 日增減%", "short_ratio": "券資比",
    "rule_score": "規則分數（5日）", "rule_score20": "規則分數（20日）",
    "rsi6": "RSI(6)", "pv_div": "量縮上漲（量價背離）",
    "mkt_regime": "大盤多空格局", "mkt_mrsi6": "大盤月線 RSI(6)", "mkt_wrsi6": "大盤週線 RSI(6)",
    "mkt_bear_div": "大盤月線空頭背離", "mkt_vol_ratio": "大盤波動度暴增", "mkt_high_vol": "大盤高波動日",
    "us_ret1": "前一晚美股 S&P 500", "sox_ret1": "前一晚費城半導體", "tsm_ret1": "前一晚台積電 ADR",
    "tsm_ret5": "台積電 ADR 近 5 日", "vix": "VIX 恐慌指數", "vix_chg5": "VIX 5 日變化",
    "chip_fx5": "外資大盤 5 日買賣超", "chip_fx20": "外資大盤 20 日買賣超", "chip_inst5": "三大法人大盤 5 日買賣超",
    "chip_txoi": "外資台指期淨未平倉", "chip_txoi_chg5": "外資台指期淨單 5 日變化",
    "chip_margin5": "大盤融資餘額 5 日增減", "chip_margin20": "大盤融資餘額 20 日增減",
    "chip_usd5": "台幣匯率 5 日變化（正＝貶值）", "chip_usd20": "台幣匯率 20 日變化",
    **{f"sub_{k}": f"規則：{k}" for k in T.KINDS},
}


# ================================================================ 特徵
def build_features(px: pd.DataFrame, mkt: pd.DataFrame | None, inst: pd.DataFrame | None,
                   mg: pd.DataFrame | None, rule: pd.DataFrame | None, ext: dict | None = None) -> pd.DataFrame:
    """每一列只用『當天收盤以前』的資料。
    ext：{"regime": 大盤環境 DataFrame, "us": {"^GSPC": 日K, "^SOX": 日K, "TSM": 日K, "^VIX": 日K}}
    美股資料以「美股日期 ≤ 台股日期」對齊：台股第 t 天的列，用美股第 t 天收盤（台灣時間 t+1 清晨）。
    台灣時間 16:30 更新時那筆還沒出現，就沿用前一天（不偷看）；06:30 更新時會補上。"""
    c, o, h, l, v = (px[k].astype(float) for k in ("Close", "Open", "High", "Low", "Volume"))
    f = pd.DataFrame(index=px.index)
    for n in (1, 5, 20, 60):
        f[f"ret{n}"] = c.pct_change(n)
    ma = {n: c.rolling(n).mean() for n in (5, 20, 60)}
    for n in (5, 20, 60):
        f[f"bias{n}"] = c / ma[n] - 1
    atr = pd.Series(T.atr(px), index=px.index)
    f["atr_pct"] = atr / c
    v20 = v.rolling(20).mean().replace(0, np.nan)
    f["vol_ratio"] = v / v20
    f["vol5_20"] = v.rolling(5).mean() / v20
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    f["pos20"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    f["dist_hi250"] = c / h.rolling(250, min_periods=60).max() - 1
    f["dist_lo250"] = c / l.rolling(250, min_periods=60).min() - 1
    f["ma20_slope"] = ma[20].pct_change(5)
    f["ma60_slope"] = ma[60].pct_change(10)
    f["gap"] = o / c.shift() - 1
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["upper"] = (h - np.maximum(o, c)) / rng
    f["lower"] = (np.minimum(o, c) - l) / rng
    up6, dn6 = d.clip(lower=0).rolling(6).mean(), (-d.clip(upper=0)).rolling(6).mean()
    f["rsi6"] = 100 * up6 / (up6 + dn6).replace(0, np.nan)
    f["pv_div"] = ((f["ret5"] > 0) & (f["vol5_20"] < 0.8)).astype(float)

    ext = ext or {}
    reg = ext.get("regime")
    if reg is not None and not reg.empty:
        rr_ = reg.reindex(px.index, method="ffill")
        f["mkt_regime"] = rr_["regime"]
        f["mkt_mrsi6"] = rr_["mrsi6"]
        f["mkt_wrsi6"] = rr_["wrsi6"]
        f["mkt_bear_div"] = rr_["bear_div"]
        f["mkt_vol_ratio"] = rr_["vol_ratio"]
        f["mkt_high_vol"] = rr_["high_vol"]
    else:
        for k in ("mkt_regime", "mkt_mrsi6", "mkt_wrsi6", "mkt_bear_div", "mkt_vol_ratio", "mkt_high_vol"):
            f[k] = 0.0
    us = ext.get("us") or {}

    def us_close(sym):
        u = us.get(sym)
        if u is None or u.empty:
            return None
        return u["Close"].astype(float).reindex(px.index, method="ffill")
    for sym, key in (("^GSPC", "us_ret1"), ("^SOX", "sox_ret1"), ("TSM", "tsm_ret1")):
        s_ = us_close(sym)
        f[key] = s_.pct_change(fill_method=None) if s_ is not None else 0.0
    s_ = us_close("TSM")
    f["tsm_ret5"] = s_.pct_change(5, fill_method=None) if s_ is not None else 0.0
    # ---- 大盤層級籌碼（台股）：都換成「相對近一年的 Z 分數」，不同年代的金額大小才能比較
    mac = ext.get("macro")

    def z(x, win=250):
        return (x - x.rolling(win, min_periods=60).mean()) / x.rolling(win, min_periods=60).std()
    if mac is not None and not mac.empty:
        m = mac.reindex(mac.index.union(px.index)).sort_index()
        g = lambda k: m[k] if k in m else pd.Series(np.nan, index=m.index)
        fx, inst_ = g("tw_foreign_net").fillna(0), g("tw_inst_net").fillna(0)
        oi, mg_, usd = g("tx_foreign_oi").ffill(), g("tw_margin_money").ffill(), g("usdtwd").ffill()
        cf = pd.DataFrame({
            "chip_fx5": z(fx.rolling(5).sum()), "chip_fx20": z(fx.rolling(20).sum()),
            "chip_inst5": z(inst_.rolling(5).sum()),
            "chip_txoi": z(oi), "chip_txoi_chg5": z(oi.diff(5)),
            "chip_margin5": mg_.pct_change(5, fill_method=None), "chip_margin20": mg_.pct_change(20, fill_method=None),
            "chip_usd5": usd.pct_change(5, fill_method=None), "chip_usd20": usd.pct_change(20, fill_method=None),
        }, index=m.index)
        cf = cf.reindex(px.index, method="ffill")
        for k in CHIP_COLS:
            f[k] = cf[k]
    else:
        for k in CHIP_COLS:
            f[k] = 0.0

    s_ = us_close("^VIX")
    f["vix"] = s_ if s_ is not None else 0.0
    f["vix_chg5"] = s_.pct_change(5, fill_method=None) if s_ is not None else 0.0

    if mkt is not None and not mkt.empty:
        mc = mkt["Close"].reindex(px.index).ffill()
        f["mkt_ret5"] = mc.pct_change(5)
        f["mkt_ret20"] = mc.pct_change(20)
        f["mkt_bias20"] = mc / mc.rolling(20).mean() - 1
        f["rel20"] = f["ret20"] - f["mkt_ret20"]
    else:
        for k in ("mkt_ret5", "mkt_ret20", "mkt_bias20", "rel20"):
            f[k] = 0.0

    avgv = (v / 1000).rolling(20).mean().replace(0, np.nan)     # 張
    if inst is not None and not inst.empty:
        ii = inst.reindex(px.index).fillna(0)
        f["inst5"] = ii["合計"].rolling(5).sum() / avgv
        f["inst20"] = ii["合計"].rolling(20).sum() / avgv
        f["foreign5"] = ii["外資"].rolling(5).sum() / avgv
        f["trust5"] = ii["投信"].rolling(5).sum() / avgv
    else:
        for k in ("inst5", "inst20", "foreign5", "trust5"):
            f[k] = 0.0
    if mg is not None and not mg.empty:
        mm = mg.reindex(px.index).ffill()
        f["margin_chg5"] = mm["融資餘額"].pct_change(5)
        f["short_ratio"] = mm["券資比%"] / 100
    else:
        f["margin_chg5"] = 0.0
        f["short_ratio"] = 0.0

    cols = [f"sub_{k}" for k in T.KINDS]
    if rule is not None and not rule.empty:
        rr = rule.reindex(px.index)
        for k in cols:
            f[k] = rr[k] if k in rr else 0.0
        f["rule_score"] = rr.get("score")
        f["rule_score20"] = rr.get("score20")
    else:
        for k in cols + ["rule_score", "rule_score20"]:
            f[k] = 0.0
    return f.replace([np.inf, -np.inf], np.nan)


# 依回測結果決定各週期使用的特徵（見 README「改版命中率比較」）
US_COLS = ["us_ret1", "sox_ret1", "tsm_ret1", "tsm_ret5", "vix", "vix_chg5"]
REG_COLS = ["mkt_regime", "mkt_mrsi6", "mkt_wrsi6", "mkt_bear_div", "mkt_vol_ratio", "mkt_high_vol"]
NEW_TECH = ["rsi6", "pv_div"]
CHIP_COLS = ["chip_fx5", "chip_fx20", "chip_inst5", "chip_txoi", "chip_txoi_chg5",
             "chip_margin5", "chip_margin20", "chip_usd5", "chip_usd20"]
EXCLUDE = {5: set(), 20: set(REG_COLS + NEW_TECH)}   # 20 日：大盤格局交給規則濾網，ML 只加美股特徵


def feature_cols(feat: pd.DataFrame, h: int, chips: bool = True) -> list[str]:
    ex = EXCLUDE.get(h, set()) | (set() if chips else set(CHIP_COLS))
    return [c for c in feat.columns if c not in ex]


def oos_score(res: dict, y: pd.Series, days: int = 250) -> dict:
    """樣本外成績：方向命中率、IC（最近 days 天、答案已揭曉的部分）。"""
    d = res["oos"][["ml"]].join(y.rename("y")).dropna().iloc[-days:]
    if len(d) < 60:
        return {"n": len(d), "dir": np.nan, "ic": np.nan}
    m = d["ml"].abs() >= 0.25
    hit = float((np.sign(d.loc[m, "ml"]) == np.sign(d.loc[m, "y"])).mean()) if m.sum() else np.nan
    return {"n": len(d), "dir": hit, "ic": float(np.corrcoef(d["ml"], d["y"])[0, 1])}


CORE = ["ret60", "bias60", "atr_pct", "vol_ratio", "rsi14", "dist_hi250", "ma60_slope"]


def labels(px: pd.DataFrame, h: int) -> pd.Series:
    c = px["Close"].astype(float)
    atr = pd.Series(T.atr(px), index=px.index)
    return (c.shift(-h) - c) / atr


# ================================================================ 模型
class Ridge:
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.ym = y.mean()
        k = int(len(Z) * 0.8)                         # 用訓練期最後 20% 挑正則化強度
        best, best_a = np.inf, 10.0
        for a in (1.0, 10.0, 100.0, 1000.0):
            w = self._solve(Z[:k], y[:k] - y[:k].mean(), a)
            e = np.mean((Z[k:] @ w + y[:k].mean() - y[k:]) ** 2) if len(Z) - k > 10 else 0
            if e < best:
                best, best_a = e, a
        self.alpha = best_a
        self.w = self._solve(Z, y - self.ym, best_a)
        return self

    @staticmethod
    def _solve(Z, y, a):
        F = Z.shape[1]
        return np.linalg.solve(Z.T @ Z + a * np.eye(F), Z.T @ y)

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.ym


class GBDT:
    """梯度提升決策樹（平方誤差、直方圖分割、隨機抽樣）。"""

    def __init__(self, n_trees=60, depth=3, lr=0.05, min_leaf=40, bins=16, subsample=0.8, seed=0):
        self.n_trees, self.depth, self.lr = n_trees, depth, lr
        self.min_leaf, self.bins, self.subsample = min_leaf, bins, subsample
        self.rng = np.random.default_rng(seed)

    def _bin(self, X):
        B = np.empty(X.shape, dtype=np.int64)
        for j in range(X.shape[1]):
            B[:, j] = np.searchsorted(self.edges[j], X[:, j], side="right")
        return B

    def fit(self, X, y):
        n, F = X.shape
        qs = np.linspace(0, 1, self.bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(F)]
        B = self._bin(X)
        self.base = float(y.mean())
        pred = np.full(n, self.base)
        self.trees, self.gain = [], np.zeros(F)
        offs = np.arange(F) * self.bins
        for _ in range(self.n_trees):
            g = y - pred
            idx = self.rng.choice(n, int(n * self.subsample), replace=False) if self.subsample < 1 else np.arange(n)
            feat, thr, left, right, val = [], [], [], [], []

            def build(ix, dep):
                nid = len(feat)
                feat.append(-1); thr.append(0); left.append(-1); right.append(-1); val.append(0.0)
                gi = g[ix]
                if dep >= self.depth or len(ix) < 2 * self.min_leaf:
                    val[nid] = self.lr * gi.mean()
                    return nid
                flat = (B[ix] + offs).ravel()
                S = np.bincount(flat, weights=np.repeat(gi, F), minlength=F * self.bins).reshape(F, self.bins)
                N = np.bincount(flat, minlength=F * self.bins).reshape(F, self.bins)
                SL, NL = S.cumsum(1)[:, :-1], N.cumsum(1)[:, :-1]
                St, Nt = gi.sum(), len(ix)
                SR, NR = St - SL, Nt - NL
                with np.errstate(divide="ignore", invalid="ignore"):
                    gain = SL ** 2 / NL + SR ** 2 / NR - St ** 2 / Nt
                gain[(NL < self.min_leaf) | (NR < self.min_leaf)] = -np.inf
                j, b = np.unravel_index(np.argmax(gain), gain.shape)
                if not np.isfinite(gain[j, b]) or gain[j, b] <= 0:
                    val[nid] = self.lr * gi.mean()
                    return nid
                self.gain[j] += gain[j, b]
                m = B[ix, j] <= b
                feat[nid], thr[nid] = int(j), int(b)
                left[nid] = build(ix[m], dep + 1)
                right[nid] = build(ix[~m], dep + 1)
                return nid

            build(idx, 0)
            tree = tuple(np.array(a) for a in (feat, thr, left, right, val))
            self.trees.append(tree)
            pred += self._tree_predict(tree, B)
        return self

    def _tree_predict(self, tree, B):
        feat, thr, left, right, val = tree
        node = np.zeros(len(B), dtype=np.int64)
        rows = np.arange(len(B))
        for _ in range(self.depth + 1):
            f = feat[node]
            leaf = f < 0
            go_left = B[rows, np.where(leaf, 0, f)] <= thr[node]
            node = np.where(leaf, node, np.where(go_left, left[node], right[node]))
        return val[node]

    def predict(self, X):
        B = self._bin(X)
        out = np.full(len(B), self.base)
        for t in self.trees:
            out += self._tree_predict(t, B)
        return out


# ================================================================ 前推驗證
def _prep(feat: pd.DataFrame):
    ok = feat[CORE].notna().all(axis=1)
    X = feat[ok].fillna(0.0)
    return X


def walk_forward(feat: pd.DataFrame, y: pd.Series, h: int, step: int = 40, n_oos: int = 500,
                 min_train: int = 300) -> dict | None:
    X = _prep(feat)
    if len(X) < min_train + 60:
        return None
    yv = y.reindex(X.index)
    Xa, ya = X.values.astype(float), yv.values.astype(float)
    N = len(Xa)
    start = max(min_train + h, N - n_oos)
    oos = np.full((N, 2), np.nan)
    for b in range(start, N, step):
        tr_end = b - h                                   # 第 b 天時，答案已知的最後一列
        m = ~np.isnan(ya[:tr_end])
        if m.sum() < min_train:
            continue
        Xt, yt = Xa[:tr_end][m], np.clip(ya[:tr_end][m], -8, 8)
        r = Ridge().fit(Xt, yt)
        g = GBDT(seed=b).fit(Xt, yt)
        e = min(b + step, N)
        oos[b:e, 0] = r.predict(Xa[b:e])
        oos[b:e, 1] = g.predict(Xa[b:e])
    # 最終模型：用所有已知答案的資料訓練，預測今天
    m = ~np.isnan(ya)
    Xt, yt = Xa[m], np.clip(ya[m], -8, 8)
    r = Ridge().fit(Xt, yt)
    g = GBDT(seed=7).fit(Xt, yt)
    today = {"ridge": float(r.predict(Xa[-1:])[0]), "gbdt": float(g.predict(Xa[-1:])[0])}
    today["ml"] = (today["ridge"] + today["gbdt"]) / 2

    df = pd.DataFrame(oos, index=X.index, columns=["ridge", "gbdt"])
    df["ml"] = df.mean(axis=1, skipna=False)
    df = df.dropna(how="all")
    done = df.join(yv.rename("y")).dropna()
    resid_sd = float((done["y"] - done["ml"]).iloc[-250:].std()) if len(done) > 30 else 1.5

    # 特徵重要度：GBDT 分割增益 ＋ 脊迴歸標準化係數，各自正規化後平均
    gi = g.gain / (g.gain.sum() or 1)
    ri = np.abs(r.w) / (np.abs(r.w).sum() or 1)
    imp = pd.Series((gi + ri) / 2, index=X.columns).sort_values(ascending=False)
    direction = pd.Series(np.sign(r.w), index=X.columns)
    return {"oos": df, "today": today, "resid_sd": resid_sd, "importance": imp, "coef_sign": direction,
            "n_train": int(m.sum()), "last_date": X.index[-1]}
