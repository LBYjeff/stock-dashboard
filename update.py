"""命令列每日更新：python update.py   （start.bat 開啟看盤前會自動執行）
只更新超過 6 小時沒更新的股票；加上 --all 則全部重跑。
加上 --rank 會另外更新排行榜股票池（universe.json），用精簡模式；
環境變數 MAX_MINUTES 可設定時間上限（雲端用），超過時剩下的股票留到下次（最久沒更新的優先）。"""
import os
import sys
import time

import pipeline as P
import store as S


def main():
    cfg = S.load_cfg()
    watch = S.all_codes(cfg)
    codes = watch if "--all" in sys.argv else [c for c in watch if P.needs_update(c)]
    lite = []
    if "--rank" in sys.argv:
        uni = S.load_universe()
        pool = [c for k in ("台股", "美股") for c in uni[k] if c not in watch]
        pool.sort(key=lambda c: S.get_meta(c).get("last_update") or "")      # 最久沒更新的先做
        lite = [c for c in pool if "--all" in sys.argv or P.needs_update(c)]
    todo = codes + lite
    if not todo:
        print("所有股票都是最新資料。")
        return
    mm = float(os.environ.get("MAX_MINUTES") or 0)
    deadline = time.time() + mm * 60 if mm else None
    print(f"更新自選股 {len(codes)} 檔、排行榜股票池 {len(lite)} 檔"
          + (f"（時間上限 {mm:.0f} 分鐘）" if mm else ""), flush=True)
    token = os.environ.get("FINMIND_TOKEN") or cfg.get("finmind_token", "")   # 雲端版從 GitHub Secrets 讀取
    res = P.update_many(todo, token, lite_codes=lite, deadline=deadline,
                        progress=lambda i, n, c: print(f"[{i + 1}/{n}] {c} ...", flush=True))
    ok = sum(1 for r in res if r.get("ok"))
    skipped = sum(1 for r in res if r.get("skipped"))
    for r in res:
        if r.get("ok"):
            ch = r["champion"]
            print(f"  ✓ {r['code']}  新增 {r['new_rows']} 筆  {r['sec']}s  "
                  f"最佳模型：5日 {P.MODEL_NAME.get(ch.get(5), '-')}／20日 {P.MODEL_NAME.get(ch.get(20), '-')}")
        elif not r.get("skipped"):
            print(f"  ✗ {r['code']}  {r.get('msg', '')}")
    print(f"完成 {ok} 檔，失敗 {len(res) - ok - skipped} 檔，留待下次 {skipped} 檔")
    S.vacuum()


if __name__ == "__main__":
    main()
