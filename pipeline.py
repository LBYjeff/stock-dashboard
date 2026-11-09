"""每日更新流程（每檔股票）：
  1. 增量抓資料：日 K、大盤、法人、融資券、股利 → market.db
  2. 規則模型：補算新日期的回溯預測
  3. 校正模型：重新校正個股權重與幅度係數
  4. 機器學習：前推驗證重新訓練，得到樣本外預測與今日預測
  5. 三模型比較：最近 250 個已驗證交易日，依「落差≤5% 命中率」（同分看 IC）選出冠軍
  6. 今日預測：三個模型＋冠軍都存進 forecast，並寫入實際紀錄 live（之後自動對答案）
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd

import data as D
import evaluate as EV
import ml
import regime as RG
import store as S
import trend as T

MODELS = ("rule", "calib", "ml")
MODEL_NAME = {"rule": "規則模型", "calib": "校正模型", "ml": "機器學習", "champion": "最佳模型"}
COMPARE_DAYS = 250
EXTRA_US = ("^GSPC", "^SOX", "^VIX", "TSM")     # 前一晚美股、費半、VIX、台積電 ADR
_reg_cache: dict = {}


def regime_for(msym: str) -> pd.DataFrame:
    """大盤環境（同一次更新中同一個指數只算一次）。"""
    px = S.load_prices(msym)
    key = (msym, str(px.index[-1]) if len(px) else "")
    if key not in _reg_cache:
        vix = S.load_prices("^VIX")
        _reg_cache[key] = RG.market_regime(px, vix if not vix.empty else None)
    return _reg_cache[key]


def us_ext() -> dict:
    return {k: S.load_prices(k) for k in EXTRA_US}


def _reg_on(reg: pd.DataFrame | None, idx) -> pd.DataFrame | None:
    if reg is None or reg.empty:
        return None
    return reg.reindex(idx, method="ffill")
_tw_info_cache: dict = {}


def tw_info(token: str = "") -> dict:
    global _tw_info_cache
    if not _tw_info_cache:
        df = D.tw_stock_info(token)
        _tw_info_cache = {} if df.empty else {r.stock_id: (r.stock_name, r.type) for r in df.itertuples()}
    return _tw_info_cache


def ensure_meta(code: str, token: str = "") -> dict:
    m = S.get_meta(code)
    if m.get("name"):
        return m
    if D.is_tw(code):
        name, mtype = tw_info(token).get(code, (code, None))
    else:
        name, mtype = D.us_name(code), None
    S.set_meta(code, name=name, mtype=mtype)
    return S.get_meta(code)


# ================================================================ 模型預測組裝
def _dir_from_move(move_atr, thr=0.25):
    return np.where(np.abs(move_atr) >= thr, np.sign(move_atr), 0)


def model_rec(base: pd.DataFrame, model: str, cal: dict, oos: dict, reg: pd.DataFrame | None = None,
              abstain: bool = True) -> pd.DataFrame:
    """把規則回溯紀錄 base 轉成指定模型的預測紀錄（同格式，可直接用 evaluate 的函式）。
    reg：大盤環境 → 規則／校正模型的逆勢訊號打折；abstain：大盤高波動日改為「觀望」。"""
    if base.empty:
        return base
    if model == "rule" and reg is None:
        return base
    r = base.copy()
    rg = _reg_on(reg, r.index)
    for h in EV.HS:
        atr, cl = r["atr"].values, r["close"].values
        if model == "rule":
            col = "score" if h == T.HORIZON else "score20"
            sc = RG.damp(r[col].values, rg["regime"].values, RG.DAMP5 if h == T.HORIZON else RG.DAMP20)
            changed = np.abs(sc - r[col].values) > 1e-9
            move = np.where(changed, EV._proj_move(sc, atr, h), r[f"p{h}"].values - cl)
            band = (r[f"hi{h}"].values - r[f"lo{h}"].values) / 2
            r[f"dirsig{h}"] = np.where(np.abs(sc) >= 2, np.sign(sc), 0)
        elif model == "calib":
            c = cal.get(h)
            if not c:
                return pd.DataFrame()
            w = np.array([c["best"][k] for k in T.KINDS])
            sc = r[[f"sub_{k}" for k in T.KINDS]].values @ w
            if rg is not None:
                sc = RG.damp(sc, rg["regime"].values, RG.DAMP5 if h == T.HORIZON else RG.DAMP20)
            move = EV._proj_move(sc, atr, h, c["best"].get("幅度", 1.0))
            band = 0.9 * atr * np.sqrt(h)
            r[f"dirsig{h}"] = np.where(np.abs(sc) >= 2, np.sign(sc), 0)
        else:
            o = oos.get(h)
            if o is None or o.empty:
                return pd.DataFrame()
            yhat = o["ml"].reindex(r.index).values
            move = yhat * atr
            sd = cal.get(f"sd{h}", 1.5)
            band = 1.28 * sd * atr                      # 約 80% 區間
            r[f"dirsig{h}"] = _dir_from_move(yhat)
        r[f"p{h}"] = cl + move
        r[f"lo{h}"], r[f"hi{h}"] = r[f"p{h}"] - band, r[f"p{h}"] + band
        if abstain and rg is not None:
            hv = rg["high_vol"].fillna(0).values > 0
            widen_on_vol(r, h, hv)
    if model == "ml":
        r = r[r["p5"].notna() | r["p20"].notna()]
    return EV.add_results(r)


def widen_on_vol(r: pd.DataFrame, h: int, mask) -> None:
    """大盤高波動日：預估價與方向不變，可能區間放寬 1.5 倍並標記「高波動」。
    （回測顯示高波動日強制觀望反而降低命中率，所以只放寬區間、提醒風險。）"""
    mask = np.asarray(mask, bool)
    r[f"hv{h}"] = mask.astype(int)
    if mask.any():
        half = (r[f"hi{h}"].values - r[f"lo{h}"].values) / 2 * 1.5
        p = r[f"p{h}"].values
        r.loc[mask, f"lo{h}"] = p[mask] - half[mask]
        r.loc[mask, f"hi{h}"] = p[mask] + half[mask]


def apply_abstain(r: pd.DataFrame, h: int, mask) -> None:
    """觀望：不給方向，預估價 = 收盤價，區間放寬 1.5 倍。"""
    mask = np.asarray(mask, bool)
    if f"abst{h}" not in r:
        r[f"abst{h}"] = 0
    if not mask.any():
        return
    cl = r["close"].values
    half = (r[f"hi{h}"].values - r[f"lo{h}"].values) / 2 * 1.5
    r.loc[mask, f"p{h}"] = cl[mask]
    r.loc[mask, f"lo{h}"] = cl[mask] - half[mask]
    r.loc[mask, f"hi{h}"] = cl[mask] + half[mask]
    if f"dirsig{h}" not in r:
        sc = r["score" if h == T.HORIZON else "score20"].values
        r[f"dirsig{h}"] = np.where(np.abs(sc) >= 2, np.sign(sc), 0)
    r.loc[mask, f"dirsig{h}"] = 0
    r.loc[mask, f"abst{h}"] = 1


def metrics(rec: pd.DataFrame, h: int, days: int = COMPARE_DAYS) -> dict:
    d = rec[rec[f"act{h}"].notna() & rec[f"p{h}"].notna()].iloc[-days:] if not rec.empty else rec
    if d is None or len(d) < 30:
        return {"n": 0}
    move = (d[f"act{h}"] / d["close"] - 1).values
    pm = (d[f"p{h}"] / d["close"] - 1).values
    dd = d[d[f"dirp{h}"] != 0]
    ic = float(np.corrcoef(pm, move)[0, 1]) if np.std(pm) > 0 else 0.0
    act = d[d[f"abst{h}"] == 0] if f"abst{h}" in d else d
    return {"n": len(d), "ok5": float(d[f"ok{h}"].mean()), "band": float(d[f"band{h}"].mean()),
            "naive": float(d[f"naive{h}"].mean()),
            "dir": float(dd[f"dir{h}"].mean()) if len(dd) else None, "n_dir": len(dd), "ic": ic,
            "n_act": len(act), "ok5_act": float(act[f"ok{h}"].mean()) if len(act) else None}


def pick_champion(ms: dict) -> str:
    cand = {m: v for m, v in ms.items() if v.get("n", 0) >= 60}
    if not cand:
        return "rule"
    best = max(cand.values(), key=lambda v: v["ok5"])["ok5"]
    close = {m: v for m, v in cand.items() if v["ok5"] >= best - 0.005}   # 0.5 個百分點內視為同分，比 IC
    return max(close, key=lambda m: (close[m]["ic"], m == "rule"))


# ================================================================ 單檔更新
def update(code: str, token: str = "", markets_done: set | None = None, log=print) -> dict:
    t0 = time.time()
    meta = ensure_meta(code, token)
    mtype = meta.get("mtype")
    msym = D.market_symbol(code)
    for sym in (msym,) + EXTRA_US:                     # 大盤指數＋前一晚美股／費半／VIX／台積電 ADR
        if markets_done is None or sym not in markets_done:
            try:
                S.update_prices(sym, None, token)
            except Exception as e:
                log(f"  {sym} 略過：{e}")
            if markets_done is not None:
                markets_done.add(sym)
    n_new = S.update_prices(code, mtype, token)
    px = S.load_prices(code)
    if px.empty or len(px) < 160:
        S.set_meta(code, last_update=S._now())
        return {"code": code, "ok": False, "msg": "資料不足或代號錯誤"}
    for fn, args in ((S.update_inst, (code, token)), (S.update_margin, (code, token)),
                     (S.update_divs, (code, mtype))):
        try:
            fn(*args)
        except Exception as e:                       # 籌碼/股利抓不到不影響主流程
            log(f"  {code} {fn.__name__} 略過：{e}")

    # 2) 規則
    S.update_rule_rec(code, px, lambda d: T.analyze(d, light=True))
    base = S.load_rule_rec(code, px)

    reg = regime_for(msym)

    # 3) 校正（用最近 500 日）
    cal = {h: EV.calibrate(base.iloc[-500:], h) for h in EV.HS}
    EV.save_calib(code, cal[5], cal[20])

    # 4) 機器學習
    mkt = S.load_prices(msym)
    inst, mg = S.load_inst(code), S.load_margin(code)
    macro = S.load_macro() if D.is_tw(code) else pd.DataFrame()
    feat = ml.build_features(px, mkt, inst, mg, base, {"regime": reg, "us": us_ext(), "macro": macro})
    ml_res, oos, chip_ab = {}, {}, {}
    for h in EV.HS:
        y = ml.labels(px, h)
        res = ml.walk_forward(feat[ml.feature_cols(feat, h, chips=False)], y, h)
        if not macro.empty and feat[ml.CHIP_COLS].notna().sum().min() > 300:
            # 大盤籌碼特徵：同一檔股票各訓練一次「有／沒有」，樣本外比較後才決定要不要用
            res_c = ml.walk_forward(feat[ml.feature_cols(feat, h, chips=True)], y, h)
            a = ml.oos_score(res, y) if res else {"dir": np.nan, "ic": np.nan}
            b = ml.oos_score(res_c, y) if res_c else {"dir": np.nan, "ic": np.nan}
            gain = np.nan_to_num(b["ic"] - a["ic"]) + np.nan_to_num(b["dir"] - a["dir"])
            use = bool(res_c) and bool(gain > 0.01)
            chip_ab[h] = {"without": a, "with": b, "used": use}
            if use:
                res = res_c
        if res:
            ml_res[h] = res
            oos[h] = res["oos"]
            S.save_ml_oos(code, h, res["oos"])
    cal_ml = dict(cal)
    for h in EV.HS:
        cal_ml[f"sd{h}"] = ml_res[h]["resid_sd"] if h in ml_res else 1.5

    # 5) 比較選冠軍
    recs = {m: model_rec(base, m, cal_ml, oos, reg) for m in MODELS}
    for h in EV.HS:
        ms = {m: metrics(recs[m], h) for m in MODELS}
        champ = pick_champion(ms)
        st = {"champion": champ, "metrics": ms,
              "calib": cal.get(h), "n_train": ml_res[h]["n_train"] if h in ml_res else 0,
              "resid_sd": cal_ml[f"sd{h}"], "chip_ab": chip_ab.get(h)}
        if h in ml_res:
            imp = ml_res[h]["importance"].head(10)
            st["importance"] = [{"key": k, "name": ml.FEATURE_NAMES.get(k, k), "w": float(v),
                                 "sign": int(ml_res[h]["coef_sign"][k])} for k, v in imp.items()]
        S.save_state(code, h, st)

    # 6) 今日預測
    last = px.index[-1]
    close = float(px["Close"].iloc[-1])
    w5, w20 = EV.weights_for(code, True)
    rg_now = _reg_on(reg, px.index[-1:])
    r_now = float(rg_now["regime"].iloc[-1]) if rg_now is not None else None
    hv_now = bool(rg_now["high_vol"].iloc[-1]) if rg_now is not None else False
    fr = T.analyze(px, regime=r_now)
    fc = T.analyze(px, w5=w5, w20=w20, regime=r_now) if (w5 or w20) else fr
    fc_all = {}
    for name, tr in (("rule", fr), ("calib", fc)):
        if tr:
            fc_all[name] = {"close": close, "atr": tr["atr"], "p5": tr["p5"], "lo5": tr["lo5"], "hi5": tr["hi5"],
                            "p20": tr["p20"], "lo20": tr["lo20"], "hi20": tr["hi20"],
                            "dir5": int(np.sign(tr["score"])) if abs(tr["score"]) >= 2 else 0,
                            "dir20": int(np.sign(tr["score20"])) if abs(tr["score20"]) >= 2 else 0,
                            "score": tr["score"], "score20": tr["score20"]}
    if ml_res and fr:
        atr = fr["atr"]
        f = {"close": close, "atr": atr}
        for h in EV.HS:
            y = ml_res[h]["today"]["ml"] if h in ml_res else 0.0
            p = close + y * atr
            b = 1.28 * cal_ml[f"sd{h}"] * atr
            f.update({f"p{h}": p, f"lo{h}": p - b, f"hi{h}": p + b, f"dir{h}": int(_dir_from_move(y)),
                      f"y{h}": y})
        f["score"] = float(np.clip(f["y5"] * 4, -8, 8))
        f["score20"] = float(np.clip(f["y20"] * 3, -8, 8))
        fc_all["ml"] = f
    for m, f in fc_all.items():
        f["regime"] = r_now
        f["high_vol"] = hv_now
        if hv_now:                                       # 高波動：區間放寬
            for h in EV.HS:
                half = (f[f"hi{h}"] - f[f"lo{h}"]) / 2 * 1.5
                f[f"lo{h}"], f[f"hi{h}"] = f[f"p{h}"] - half, f[f"p{h}"] + half
        S.save_forecast(code, m, last, f)
        S.log_live(code, last, m, f)
    # 冠軍（5 日、20 日可以是不同模型）
    champ = {}
    for h in EV.HS:
        cm = S.load_state(code, h).get("champion", "rule")
        cm = cm if cm in fc_all else "rule"
        src = fc_all.get(cm) or fc_all.get("rule")
        if src:
            champ.update({k: src[k] for k in (f"p{h}", f"lo{h}", f"hi{h}", f"dir{h}")})
            champ[f"model{h}"] = cm
    if champ and fc_all:
        champ.update({"close": close, "atr": fc_all["rule"]["atr"], "regime": r_now, "high_vol": hv_now})
        champ["score"] = fc_all[champ["model5"]]["score"]
        champ["score20"] = fc_all[champ["model20"]]["score20"]
        S.save_forecast(code, "champion", last, champ)
        S.log_live(code, last, "champion", champ)

    S.set_meta(code, last_update=S._now())
    return {"code": code, "ok": True, "new_rows": n_new, "sec": round(time.time() - t0, 1),
            "champion": {h: S.load_state(code, h).get("champion") for h in EV.HS}}


def update_many(codes: list[str], token: str = "", progress=None, log=print) -> list[dict]:
    """progress(i, n, code)：給畫面顯示進度。"""
    out, done = [], set()
    with S.LOCK:
        if any(D.is_tw(c) for c in codes):
            try:
                n = S.update_macro(token, log)
                log(f"大盤籌碼資料：新增 {n} 筆")
            except Exception as e:
                log(f"大盤籌碼資料略過：{e}")
        for i, c in enumerate(codes):
            if progress:
                progress(i, len(codes), c)
            try:
                out.append(update(c, token, done, log))
            except Exception as e:
                log(f"{c} 更新失敗：{e}")
                out.append({"code": c, "ok": False, "msg": str(e)})
    return out


def needs_update(code: str, hours: float = 6) -> bool:
    lu = S.get_meta(code).get("last_update")
    if not lu:
        return True
    return (dt.datetime.now() - dt.datetime.fromisoformat(lu)).total_seconds() > hours * 3600


CONFLICT_ABSTAIN = False    # 規則模型與機器學習方向相反時，最佳模型改為觀望


def champion_rec(recs: dict, champs: dict, conflict: bool | None = None) -> pd.DataFrame:
    """各週期採用冠軍模型的預測，組成一份紀錄；冠軍與另一模型方向相反時觀望。"""
    conflict = CONFLICT_ABSTAIN if conflict is None else conflict
    base = recs.get("rule", pd.DataFrame())
    if base.empty:
        return base
    r = base.copy()
    for h in EV.HS:
        src = recs.get(champs.get(h, "rule"))
        if src is None or src.empty:
            src = base
        src = src.reindex(r.index)
        for col in (f"p{h}", f"lo{h}", f"hi{h}"):
            r[col] = src[col]
        r[f"dirsig{h}"] = src[f"dirp{h}"]
        r[f"abst{h}"] = src[f"abst{h}"].fillna(0).values if f"abst{h}" in src else 0
        if conflict:
            other = [m for m in ("rule", "ml") if m != champs.get(h)]
            o = recs.get(other[0]) if other else None
            if o is not None and not o.empty and f"dirp{h}" in o:
                od = o[f"dirp{h}"].reindex(r.index).fillna(0).values
                cd = r[f"dirsig{h}"].fillna(0).values
                apply_abstain(r, h, (od * cd) < 0)
    r = r[r["p5"].notna()]
    return EV.add_results(r)


def model_section(states: dict) -> str:
    """初報告的「模型比較」段落。"""
    L = ["## 四、三種模型比較（最近 250 個已驗證交易日，樣本外）"]
    for h in EV.HS:
        st = states.get(h) or {}
        ms = st.get("metrics") or {}
        if not ms:
            L.append(f"- {h} 日：尚無足夠資料。")
            continue
        for m in MODELS:
            v = ms.get(m, {})
            if not v.get("n"):
                L.append(f"- {h} 日・{MODEL_NAME[m]}：樣本不足")
                continue
            L.append(f"- {h} 日・{MODEL_NAME[m]}：落差≤5% {EV.pct(v['ok5'])}（股價不動 {EV.pct(v['naive'])}）、"
                     f"落在區間 {EV.pct(v['band'])}、方向命中 {EV.pct(v.get('dir'))}（{v.get('n_dir', 0)} 次）、IC {v['ic']:+.3f}")
        L.append(f"- **{h} 日目前採用：{MODEL_NAME.get(st.get('champion'), '-')}**")
        ab = st.get("chip_ab")
        if ab:
            w, wo = ab["with"], ab["without"]
            L.append(f"  - 大盤籌碼特徵：{'✅ 採用' if ab['used'] else '❌ 不採用'}"
                     f"（樣本外方向命中 {EV.pct(wo.get('dir'))} → {EV.pct(w.get('dir'))}，"
                     f"IC {wo.get('ic') or 0:+.3f} → {w.get('ic') or 0:+.3f}）")
        imp = st.get("importance") or []
        if imp:
            L.append(f"  - 機器學習最看重：" + "、".join(
                f"{i['name']}（{'正' if i['sign'] > 0 else '負' if i['sign'] < 0 else '非線性'}相關）" for i in imp[:5]))
    L.append("")
    L.append("> 每天更新資料後，三個模型都會重新訓練並重新比較，冠軍可能隨市場狀態改變。"
             "「樣本外」代表每一筆預測都只用當時已知的資料，沒有偷看答案。")
    L.append("")
    return "\n".join(L)
