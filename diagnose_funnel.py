"""
diagnose_funnel.py — 미국 선별이 왜 N개로 줄어드는지 단계별 추적

각 깔때기(funnel) 단계에서 종목이 몇 개 살아남는지 보여준다:
  [A] 유니버스 전체 티커 수
  [B] OHLCV(가격) 수집 성공 수      ← 여기서 대량 탈락하면 yfinance 문제
  [C] 앞단 필터 통과 (가격/시총/거래대금/이력)
  [D] 팩터 계산 성공
  [E] 트랙 배정 (Avoid 제외 후 실제 후보)
  [F] 재무(fund_grade) 보유 수      ← 재무는 '필터'가 아니라 '부가정보'

실행:
  cd ~/etf-radar-run/etf-radar
  source .venv/bin/activate
  python3 diagnose_funnel.py
"""

import math
import numpy as np
import pandas as pd

GREEN="\033[92m"; RED="\033[91m"; YEL="\033[93m"; DIM="\033[2m"; CYAN="\033[96m"; END="\033[0m"
def line(label, n, note=""):
    return f"   {CYAN}{label:42s}{END} {n:>5}  {DIM}{note}{END}"

print("="*70)
print("미국 선별 깔때기(Funnel) 진단 — 어디서 종목이 줄어드는가")
print("="*70)

# ──────────────────────────────────────────────────────────────
# 0) 캐시에 이미 계산된 full_df가 있으면 그것부터 분석 (가장 빠름)
# ──────────────────────────────────────────────────────────────
from screener.engine import _load_cache, US_FILTER, AVOID_TRACKS, LEADER_TRACKS, BREAKOUT_TRACKS

print(f"\n[필터 기준] 가격≥${US_FILTER['min_price']} / "
      f"시총≥${US_FILTER['min_market_cap']/1e9:.0f}B / "
      f"거래대금≥${US_FILTER['min_avg_dv']/1e6:.0f}M / "
      f"이력≥{US_FILTER['min_history_days']}일")

cached = _load_cache("us")
if cached is not None and not cached.empty:
    print(f"\n{GREEN}● 캐시된 계산 결과(full_df) 발견 — 이걸로 분석{END}")
    df = cached
    n_total = len(df)
    print(line("[D] 팩터 계산 완료 종목 (full_df)", n_total))

    # 트랙 분포
    if "track" in df.columns:
        print(f"\n   {YEL}트랙 분포:{END}")
        vc = df["track"].value_counts()
        for tr, cnt in vc.items():
            mark = f"{RED}(Avoid→제외){END}" if tr in AVOID_TRACKS else ""
            print(f"      {str(tr):32s} {cnt:>4}  {mark}")

        n_avoid = int(df["track"].isin(AVOID_TRACKS).sum())
        n_after_avoid = n_total - n_avoid
        print(line("[E] Avoid 제외 후 남는 종목", n_after_avoid,
                   f"({n_avoid}개가 Avoid/Weak로 제외)"))

    # 재무 보유 여부 (필터가 아님을 확인)
    if "fund_grade" in df.columns:
        has_fund = df["fund_grade"].apply(lambda x: x not in (None, "", np.nan)).sum()
        no_fund  = n_total - has_fund
        print(line("[F] 재무등급(fund_grade) 보유 종목", int(has_fund),
                   f"({no_fund}개는 재무 없음 — 그래도 선별엔 남아있음)"))
        print(f"\n   {DIM}※ 재무가 없어도 종목은 선별에서 빠지지 않습니다.{END}")
        print(f"   {DIM}  재무는 점수에 가산될 뿐, 탈락 사유가 아님.{END}")

    # 화면 후보(build_candidates 유사) 추정: Avoid 제외 + Risk 극단 제외
    if "track" in df.columns:
        cand = df[~df["track"].isin(AVOID_TRACKS)].copy()
        # 앱의 build_candidates는 Risk>=90 & 음봉/윗꼬리 같은 추가 컷이 있음
        if "top_risk_score" in cand.columns and "ret_5d" in cand.columns:
            risk = cand["top_risk_score"].fillna(0)
            ret5 = cand["ret_5d"].fillna(0)
            flag = cand["risk_flag"].fillna("") if "risk_flag" in cand.columns else pd.Series("", index=cand.index)
            extreme = (risk >= 90) & (ret5 < 0) & flag.str.contains("음봉|윗꼬리", regex=True)
            n_extreme = int(extreme.sum())
            cand = cand[~extreme]
            print(line("[E'] Risk 극단 추가 제외 후", len(cand),
                       f"({n_extreme}개 추가 제외)"))
        print(f"\n{GREEN}★ 최종 화면 후보 추정치: 약 {len(cand)}개{END}")
        print(f"   {DIM}(앱의 트랙 필터/정렬에 따라 화면 표시는 더 적을 수 있음){END}")

    print("\n" + "="*70)
    print("판정")
    print("="*70)
    if "track" in df.columns:
        n_avoid = int(df["track"].isin(AVOID_TRACKS).sum())
        if n_avoid > n_total * 0.5:
            print(RED + "● 절반 이상이 'Avoid/Weak' 트랙 → 시장 약세거나 팩터 컷이 빡빡함" + END)
            print("  (재무 문제 아님. 트랙 분류 기준 문제)")
        elif n_total < 30:
            print(YEL + f"● 팩터 계산된 종목 자체가 {n_total}개로 적음" + END)
            print("  → 가격(OHLCV) 수집 단계에서 대량 실패했을 가능성 큼")
            print("  → 아래 [B] 수집 단계 점검 결과를 확인하세요")
        else:
            print(GREEN + "● 팩터 종목은 충분함. 화면이 적다면 트랙/정렬 필터 때문" + END)

else:
    print(f"\n{YEL}● 캐시된 계산 결과 없음 — 수집 단계만 점검{END}")
    print(f"  (먼저 앱에서 미국 스크리너를 한 번 돌리면 더 정밀하게 진단 가능)")

# ──────────────────────────────────────────────────────────────
# [B] OHLCV 수집 성공률 점검 (유니버스 일부 샘플)
# ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("[B] OHLCV(가격) 수집 성공률 — yfinance가 실제로 데이터를 주는가")
print("="*70)
try:
    from screener.collector import get_us_tickers
    try:
        tickers = get_us_tickers()
    except Exception:
        tickers = []
    if not tickers:
        tickers = ["AAPL","NVDA","MSFT","AMD","MU","TSLA","META","GOOGL",
                   "AMZN","AVGO","CRM","ORCL","ADBE","NFLX","INTC"]
        print(f"   {DIM}유니버스 캐시 없음 → 대표 {len(tickers)}개로 샘플 테스트{END}")
    else:
        print(f"   유니버스 티커 수: {len(tickers)}개")
        # 너무 많으면 앞 60개만 샘플
        if len(tickers) > 60:
            print(f"   {DIM}(앞 60개만 샘플 테스트){END}")
            tickers = tickers[:60]

    import yfinance as yf
    raw = yf.download(tickers, period="3mo", auto_adjust=True,
                      actions=False, progress=False, threads=True, group_by="ticker")
    ok_cnt = 0
    if raw is not None and not raw.empty and isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                d = raw[t].dropna(how="all")
                if len(d) >= 20:
                    ok_cnt += 1
            except Exception:
                pass
    elif raw is not None and not raw.empty:
        ok_cnt = 1 if len(raw.dropna(how="all")) >= 20 else 0

    rate = ok_cnt/len(tickers)*100 if tickers else 0
    print(line("샘플 중 가격 수집 성공", ok_cnt, f"/ {len(tickers)} ({rate:.0f}%)"))
    if rate < 50:
        print(RED + "   ● 절반 이상 실패 → yfinance/네트워크 문제가 선별 축소의 주원인" + END)
        print("     해결: pip install -U yfinance  (버전 갱신이 1순위)")
    elif rate < 90:
        print(YEL + "   ● 일부 실패 → 레이트리밋 가능성. 시간 두고 재시도" + END)
    else:
        print(GREEN + "   ● 가격 수집은 정상 → 선별 축소는 트랙 분류 쪽 문제" + END)
except Exception as e:
    print(RED + f"   수집 테스트 예외: {e}" + END)

print()
