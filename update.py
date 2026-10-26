"""命令列每日更新：python update.py   （start.bat 開啟看盤前會自動執行）
只更新超過 6 小時沒更新的股票；加上 --all 則全部重跑。"""
import os
import sys

import pipeline as P
import store as S


def main():
    cfg = S.load_cfg()
    codes = S.all_codes(cfg)
    if "--all" not in sys.argv:
        codes = [c for c in codes if P.needs_update(c)]
    if not codes:
        print("所有自選股都是最新資料。")
        return
    print(f"更新 {len(codes)} 檔：{', '.join(codes)}（第一次建立資料庫每檔約 10～30 秒）")
    token = os.environ.get("FINMIND_TOKEN") or cfg.get("finmind_token", "")   # 雲端版從 GitHub Secrets 讀取
    res = P.update_many(codes, token,
                        progress=lambda i, n, c: print(f"[{i + 1}/{n}] {c} ...", flush=True))
    for r in res:
        if r.get("ok"):
            ch = r["champion"]
            print(f"  ✓ {r['code']}  新增 {r['new_rows']} 筆  {r['sec']}s  "
                  f"最佳模型：5日 {P.MODEL_NAME.get(ch.get(5), '-')}／20日 {P.MODEL_NAME.get(ch.get(20), '-')}")
        else:
            print(f"  ✗ {r['code']}  {r.get('msg', '')}")


if __name__ == "__main__":
    main()
