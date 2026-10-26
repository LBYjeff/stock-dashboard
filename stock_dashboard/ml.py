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
    **{f"sub_{k}": f"規則：{k}" for k in T.KINDS},
}


# ================================================================ 特徵
def build_features(px: pd.DataFrame, mkt: pd.DataFrame | None, inst: pd.DataFrame | None,
                   mg: pd.DataFrame | None, rule: pd.DataFrame | None) -> pd.DataFrame:
    """每一列只用『當天收盤以前』的資料。"""
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
