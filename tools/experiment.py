#!/usr/bin/env python3
"""
실험 하니스 — Streamlit 앱 없이, 재수집 없이 점수/가중치를 빠르게 실험한다.

3단계 캐시 (느린 → 빠른):
  L1 snapshot  : 유니버스 수집 전체를 안정 파일로 저장  (네트워크, 느림)
                 → 새 데이터가 필요할 때만 다시.
  L2 backtest  : 종목별 점수·수익 레코드(df_records) 생성  (CPU, 네트워크 없음)
                 → 팩터 *계산 로직*(factors.py)을 바꿨을 때만 다시.
  L3 weights   : 기존 레코드로 합성점수를 재계산해 Spearman·분위 출력  (즉시)
                 → 가중치만 바꿀 때. 백테스트 재실행 불필요.

사용 예:
  # 1) 최초 1회 (느림): 데이터 수집 → 스냅샷 저장
  python3 tools/experiment.py snapshot kr

  # 2) 백테스트 레코드 생성 (팩터 로직 바꿨을 때)
  python3 tools/experiment.py backtest kr

  # 3) 진단 출력 (현재 코드 가중치로)
  python3 tools/experiment.py report kr

  # 4) 가중치만 바꿔 즉시 실험 (백테스트 재실행 X)
  python3 tools/experiment.py weights kr --w leader=0.45,volume=0.20,breakout=0.15,catalyst=0.20,risk_penalty=0.08

  # 한 번에 (없는 단계는 자동 생성):
  python3 tools/experiment.py report kr --auto
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from screener.paths import CACHE_DIR as CACHE  # zip 바깥 공용 캐시 위치
CACHE.mkdir(parents=True, exist_ok=True)


def _snap_path(market):  return CACHE / f"_snap_{market}.pkl"
def _rec_path(market, la=20):  return CACHE / f"_bt_records_{market}_la{la}.pkl"

# 시장별 초과수익 기준 컬럼
EXCESS_COL = {"kr": "excess_kospi", "us": "excess_spy"}


# ════════════════════════════════════════════════════════════════════
# L1 — 스냅샷 (네트워크)
# ════════════════════════════════════════════════════════════════════
def make_snapshot(market: str, with_supply: bool = False):
    from screener.collector import (
        get_kr_tickers, get_us_tickers,
        fetch_kr_universe, fetch_us_universe, fetch_kr_benchmarks,
        attach_kr_supply_series,
    )
    print(f"[L1] {market.upper()} 수집 시작 (네트워크 — 한국은 수십 분 걸릴 수 있음)…")
    if market == "kr":
        data, failed = fetch_kr_universe(get_kr_tickers())
        if with_supply:
            print("[L1] 수급 시계열 부착 중 (최초 1회 느림, 종목당 30페이지)…")
            n = attach_kr_supply_series(data)
            print(f"[L1] 수급 부착: {n}종목")
        bench = fetch_kr_benchmarks()
        bt_data = {**data}
        for code, close_s in bench.items():
            if code not in bt_data:
                frame = close_s.to_frame("Close") if isinstance(close_s, pd.Series) else close_s
                bt_data[code] = {"ohlcv": frame, "name": f"벤치마크_{code}", "market": "INDEX"}
        payload = {"bt_data": bt_data, "n_fail": len(failed)}
    else:
        data, failed = fetch_us_universe(get_us_tickers())
        payload = {"bt_data": data, "n_fail": len(failed)}
    with open(_snap_path(market), "wb") as f:
        pickle.dump(payload, f)
    _sup = sum(1 for v in payload["bt_data"].values()
               if isinstance(v, dict) and "supply_series" in v)
    print(f"[L1] 저장 완료: {_snap_path(market).name}  "
          f"(종목 {len(payload['bt_data'])}, 수집실패 {payload['n_fail']}, 수급부착 {_sup})")


# ════════════════════════════════════════════════════════════════════
# L2 — 백테스트 레코드 (CPU)
# ════════════════════════════════════════════════════════════════════
def make_records(market: str, lookahead: int = 20):
    if not _snap_path(market).exists():
        print(f"[L2] 스냅샷 없음 → 먼저: python3 tools/experiment.py snapshot {market}")
        sys.exit(1)
    with open(_snap_path(market), "rb") as f:
        bt_data = pickle.load(f)["bt_data"]

    from screener.backtest import (
        run_kr_walkforward_backtest, run_walkforward_backtest,
    )
    print(f"[L2] {market.upper()} 백테스트 실행 (lookahead={lookahead}일, 네트워크 없음)…")
    if market == "kr":
        max_hist = max((len(v["ohlcv"]) for v in bt_data.values() if "ohlcv" in v), default=0)
        min_hist = 120 if max_hist >= 300 else 80 if max_hist >= 160 else 60
        res = run_kr_walkforward_backtest(bt_data, lookahead=lookahead, step=5,
                                          min_history=min_hist, bench_code="kospi")
    else:
        res = run_walkforward_backtest(bt_data, lookahead=lookahead, step=5)

    df = res.get("df_records")
    if df is None or len(df) == 0:
        print("[L2] 레코드 없음 — 데이터/필터 확인 필요"); sys.exit(1)
    with open(_rec_path(market, lookahead), "wb") as f:
        pickle.dump(df, f)
    print(f"[L2] 저장 완료: {_rec_path(market, lookahead).name}  "
          f"(레코드 {len(df)}, 종목 {df['ticker'].nunique()})")


# ════════════════════════════════════════════════════════════════════
# L3 — 진단 출력 (즉시)
# ════════════════════════════════════════════════════════════════════
def _load_records(market, la=20):
    if not _rec_path(market, la).exists():
        print(f"[L3] 레코드 없음(la={la}) → 먼저: python3 tools/experiment.py backtest {market} --la {la}")
        sys.exit(1)
    with open(_rec_path(market, la), "rb") as f:
        return pickle.load(f)


def _spearman(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 10:
        return float("nan")
    return float(s["a"].corr(s["b"], method="spearman"))


def _quintile_table(df, score_col, excess_col):
    d = df[[score_col, excess_col, "ret"]].dropna(subset=[score_col]).copy()
    if len(d) < 25:
        print("   (표본 부족)"); return
    d["q"] = pd.qcut(d[score_col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    print(f"   {'분위':<6}{'n':>7}{'평균점수':>9}{'평균수익%':>10}{'초과수익%':>10}{'승률%':>8}")
    for q in [1, 2, 3, 4, 5]:
        g = d[d["q"] == q]
        if len(g) == 0:
            continue
        win = (g["ret"] > 0).mean() * 100
        print(f"   Q{q:<5}{len(g):>7}{g[score_col].mean():>9.1f}"
              f"{g['ret'].mean():>10.2f}{g[excess_col].mean():>10.2f}{win:>8.1f}")
    # 단조성 체크
    means = [df_q[excess_col].mean() for q in [1,2,3,4,5]
             for df_q in [d[d["q"] == q]] if len(df_q)]
    mono = all(means[i] <= means[i+1] for i in range(len(means)-1))
    print(f"   → 초과수익 단조 증가: {'예 ✅' if mono else '아니오 ❌ (점수가 수익을 정렬하지 못함)'}")


def report(market: str, weights: dict = None, la: int = 20):
    df = _load_records(market, la).copy()
    excess = EXCESS_COL[market]
    if excess not in df.columns:
        excess = "excess_spy"

    score_col = "composite"
    if weights:
        # L3 핵심: 기존 레코드로 합성점수만 재계산 (백테스트 재실행 없음)
        from utils.scoring import calc_validated_score_series
        df["composite_exp"] = calc_validated_score_series(df, weights=weights)
        score_col = "composite_exp"
        print(f"\n[가중치 실험] {weights}")
    else:
        if "composite" not in df.columns:
            # 레코드에 composite가 없으면 현재 코드 기본 가중치로 재계산
            from utils.scoring import calc_validated_score_series
            df["composite"] = calc_validated_score_series(df)
        print(f"\n[현재 코드 가중치로 진단]")

    print(f"시장 {market.upper()} | lookahead {la}일 | 레코드 {len(df)} | 종목 {df['ticker'].nunique()} | 기준 {excess}")
    sp_excess = _spearman(df[score_col], df[excess])
    sp_ret    = _spearman(df[score_col], df["ret"])
    print(f"Spearman(점수, {excess}) = {sp_excess:+.4f}   "
          f"Spearman(점수, 원수익) = {sp_ret:+.4f}")
    print(f"전체 평균 초과수익 = {df[excess].mean():+.2f}%   "
          f"전체 승률 = {(df['ret']>0).mean()*100:.1f}%")
    print("분위표:")
    _quintile_table(df, score_col, excess)

    # 개별 팩터 예측력 (재계산 불필요 — 레코드에 이미 있음)
    if not weights:
        print("\n개별 팩터 Spearman(초과수익):")
        cand = ["leader","volume","breakout","catalyst","quality","risk",
                "volume_enhanced","breakout_enhanced","risk_combined",
                "momentum_accel","near_high","vol_contract","pullback_q",
                "supply_flow","sect_rs"]
        rows = []
        for c in cand:
            if c in df.columns:
                rows.append((c, _spearman(df[c], df[excess])))
        rows.sort(key=lambda x: (np.nan_to_num(x[1], nan=-9)), reverse=True)
        for c, s in rows:
            tag = "🟢" if s > 0.03 else "🔴" if s < -0.02 else "⚪"
            print(f"   {tag} {c:<20}{s:+.4f}")


# ════════════════════════════════════════════════════════════════════
def sweep(market: str, horizons):
    """
    여러 보유기간(lookahead)에서 합성점수의 예측력을 비교한다.
    리레이팅(느린 멀티플 확장) 가설이 더 긴 창에서 살아나는지 확인하는 용도.
    각 기간의 백테스트 레코드가 없으면 자동 생성(네트워크 없음, 스냅샷 재사용).
    """
    excess = EXCESS_COL[market]
    print(f"\n=== {market.upper()} 보유기간 스윕 (스냅샷 재사용, 재수집 없음) ===")
    print(f"{'기간':>6}{'레코드':>8}{'Spearman(초과)':>16}{'상위20%초과%':>14}{'단조':>6}")
    for la in horizons:
        if not _rec_path(market, la).exists():
            make_records(market, lookahead=la)
        df = _load_records(market, la).copy()
        ex = excess if excess in df.columns else "excess_spy"
        if "composite" not in df.columns:
            from utils.scoring import calc_validated_score_series
            df["composite"] = calc_validated_score_series(df)
        sp = _spearman(df["composite"], df[ex])
        # 상위 20% 초과수익 + 분위 단조성
        d = df[["composite", ex]].dropna()
        q = pd.qcut(d["composite"].rank(method="first"), 5, labels=[1,2,3,4,5])
        means = [d[ex][q == i].mean() for i in [1,2,3,4,5]]
        top = means[-1]
        mono = all(means[i] <= means[i+1] for i in range(4))
        print(f"{la:>5}일{len(df):>8}{sp:>+16.4f}{top:>+14.2f}{'✅' if mono else '❌':>6}")
    print("\n해석: 더 긴 기간에서 Spearman이 +로 뚜렷해지거나 단조 ✅면,")
    print("      전략의 시간축이 20일보다 길다는 뜻 → 그 기간으로 운영 고려.")


def _parse_weights(s: str) -> dict:
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "backtest", "report", "weights", "sweep"])
    ap.add_argument("market", choices=["kr", "us"])
    ap.add_argument("--w", help="가중치 (예: leader=0.4,volume=0.2,...)")
    ap.add_argument("--la", type=int, default=20, help="보유기간(lookahead) 일수 (기본 20)")
    ap.add_argument("--horizons", default="20,60,120", help="sweep용 기간들 (예: 20,60,120)")
    ap.add_argument("--supply", action="store_true", help="snapshot 시 수급 시계열도 수집(백테스트 수급 편입용)")
    ap.add_argument("--auto", action="store_true", help="없는 단계 자동 생성")
    args = ap.parse_args()

    if args.cmd == "snapshot":
        make_snapshot(args.market, with_supply=args.supply)
    elif args.cmd == "backtest":
        make_records(args.market, lookahead=args.la)
    elif args.cmd == "report":
        if args.auto:
            if not _snap_path(args.market).exists(): make_snapshot(args.market)
            if not _rec_path(args.market, args.la).exists(): make_records(args.market, lookahead=args.la)
        report(args.market, la=args.la)
    elif args.cmd == "weights":
        if not args.w:
            print("--w 필요 (예: --w leader=0.4,volume=0.2,breakout=0.15,catalyst=0.2,risk_penalty=0.08)")
            sys.exit(1)
        report(args.market, weights=_parse_weights(args.w), la=args.la)
    elif args.cmd == "sweep":
        if not _snap_path(args.market).exists():
            print(f"스냅샷 없음 → 먼저: python3 tools/experiment.py snapshot {args.market}")
            sys.exit(1)
        sweep(args.market, [int(x) for x in args.horizons.split(",")])


if __name__ == "__main__":
    main()
