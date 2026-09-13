"""
diagnose_us.py — 미국 데이터 수집 단계별 진단

어디서 데이터가 비는지 정확히 짚어준다:
  [1] yfinance 설치/버전
  [2] OHLCV(가격) 수집      ← 이게 되면 "분석 자체"는 됨
  [3] info(메타/밸류에이션)  ← 이게 안 되면 sector/PER 등이 빔
  [4] 재무제표(분기 손익)    ← 이게 안 되면 op_growth 등이 빔
  [5] 펀더멘털 종합 점수      ← fund_score/fund_grade
  [6] 애널리스트(목표가)

실행:
  cd ~/etf-radar-run/etf-radar
  source .venv/bin/activate
  python3 diagnose_us.py
  python3 diagnose_us.py AAPL NVDA MU   # 특정 티커만
"""

import sys
import json

# 테스트 대상 (인자로 주면 그걸 쓰고, 없으면 대표 5개)
TEST_TICKERS = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "NVDA", "MSFT", "MU", "AMD"]

GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; DIM = "\033[2m"; END = "\033[0m"
def ok(s):   return f"{GREEN}✅ {s}{END}"
def bad(s):  return f"{RED}❌ {s}{END}"
def warn(s): return f"{YEL}⚠️  {s}{END}"

print("=" * 64)
print(f"미국 데이터 수집 진단 — 대상: {', '.join(TEST_TICKERS)}")
print("=" * 64)

# ──────────────────────────────────────────────────────────────
# [1] yfinance 설치/버전
# ──────────────────────────────────────────────────────────────
print("\n[1] yfinance 설치/버전")
try:
    import yfinance as yf
    ver = getattr(yf, "__version__", "unknown")
    print("   " + ok(f"yfinance {ver} 임포트 성공"))
except Exception as e:
    print("   " + bad(f"yfinance 임포트 실패: {e}"))
    print("   → pip install -U yfinance 후 다시 시도")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# [2] OHLCV(가격) 배치 다운로드  ← 분석의 핵심
# ──────────────────────────────────────────────────────────────
print("\n[2] OHLCV(가격) 수집  — 이게 되면 '분석 자체'는 작동")
price_ok = []
try:
    raw = yf.download(
        TEST_TICKERS, period="3mo",
        auto_adjust=True, actions=False,
        progress=False, threads=True, group_by="ticker",
    )
    import pandas as pd
    if raw is None or raw.empty:
        print("   " + bad("yf.download 결과가 비어있음 (전체 실패)"))
    else:
        if isinstance(raw.columns, pd.MultiIndex):
            for t in TEST_TICKERS:
                try:
                    df = raw[t].dropna(how="all")
                    n = len(df)
                    if n >= 20:
                        last = df["Close"].dropna().iloc[-1]
                        print("   " + ok(f"{t:6s} {n}일치, 최근가 {last:.2f}"))
                        price_ok.append(t)
                    else:
                        print("   " + warn(f"{t:6s} 데이터 {n}일치뿐 (20일 미만 → 제외됨)"))
                except Exception as e:
                    print("   " + bad(f"{t:6s} 추출 실패: {e}"))
        else:
            # 단일 티커
            df = raw.dropna(how="all")
            t = TEST_TICKERS[0]
            if len(df) >= 20:
                print("   " + ok(f"{t:6s} {len(df)}일치"))
                price_ok.append(t)
            else:
                print("   " + warn(f"{t:6s} 데이터 부족"))
except Exception as e:
    print("   " + bad(f"OHLCV 다운로드 예외: {e}"))

if not price_ok:
    print("\n" + bad("가격 데이터를 하나도 못 가져옴 → 네트워크/yfinance 차단 의심"))
    print("   - VPN/방화벽, 또는 yfinance 버전 문제일 수 있음")
    print("   - pip install -U yfinance 시도 권장")
else:
    print("   " + DIM + f"({len(price_ok)}/{len(TEST_TICKERS)}개 가격 수집 성공)" + END)

# ──────────────────────────────────────────────────────────────
# [3] info(메타/밸류에이션)
# ──────────────────────────────────────────────────────────────
print("\n[3] info(섹터/PER/시총 등 메타)  — 안 되면 sector·PER 등이 빔")
info_ok = []
for t in TEST_TICKERS:
    try:
        obj = yf.Ticker(t)
        raw_info = obj.info
        info = raw_info if isinstance(raw_info, dict) else {}
        keys = ["sector", "trailingPE", "forwardPE", "marketCap", "revenueGrowth"]
        present = {k: info.get(k) for k in keys if info.get(k) not in (None, "")}
        if len(info) == 0:
            print("   " + bad(f"{t:6s} info 비어있음 (yfinance가 빈 dict 반환)"))
        elif len(present) == 0:
            print("   " + warn(f"{t:6s} info는 있으나 핵심 필드 전부 None (필드 {len(info)}개)"))
        else:
            print("   " + ok(f"{t:6s} 섹터={info.get('sector','?')} / PER={info.get('trailingPE','?')} / 매출성장={info.get('revenueGrowth','?')}"))
            info_ok.append(t)
    except Exception as e:
        err = str(e).lower()
        tag = "RATE LIMIT" if ("rate" in err or "429" in err or "too many" in err) else "ERROR"
        print("   " + bad(f"{t:6s} info 예외 [{tag}]: {str(e)[:60]}"))

# ──────────────────────────────────────────────────────────────
# [4] 재무제표(분기 손익)
# ──────────────────────────────────────────────────────────────
print("\n[4] 분기 손익계산서  — 안 되면 영업이익 성장률 등이 빔")
for t in TEST_TICKERS:
    try:
        obj = yf.Ticker(t)
        q = obj.quarterly_income_stmt
        if q is None or q.empty:
            print("   " + warn(f"{t:6s} 분기 손익 비어있음"))
        else:
            print("   " + ok(f"{t:6s} 분기 {q.shape[1]}개 / 항목 {q.shape[0]}개"))
    except Exception as e:
        print("   " + bad(f"{t:6s} 손익 예외: {str(e)[:60]}"))

# ──────────────────────────────────────────────────────────────
# [5] 펀더멘털 종합 점수 (프로젝트 함수 사용)
# ──────────────────────────────────────────────────────────────
print("\n[5] 펀더멘털 종합(fund_score/fund_grade)  — 프로젝트 함수 직접 호출")
try:
    from screener.fundamental import fetch_us_fundamental
    for t in TEST_TICKERS:
        try:
            f = fetch_us_fundamental(t)
            fs = f.get("fund_score")
            fg = f.get("fund_grade")
            rg = f.get("rev_growth")
            if fs is None or fg in (None, ""):
                print("   " + bad(f"{t:6s} fund_score=None (수집 실패) — 등급 비어있음"))
            else:
                print("   " + ok(f"{t:6s} 점수={fs:.0f} 등급={fg} 매출성장={rg}"))
        except Exception as e:
            print("   " + bad(f"{t:6s} 펀더멘털 예외: {str(e)[:60]}"))
except Exception as e:
    print("   " + bad(f"fundamental 모듈 임포트 실패: {e}"))

# ──────────────────────────────────────────────────────────────
# [6] 애널리스트(목표가)
# ──────────────────────────────────────────────────────────────
print("\n[6] 애널리스트 목표가/업사이드")
try:
    from screener.analyst import fetch_analyst_batch
    res = fetch_analyst_batch(TEST_TICKERS) if price_ok else {}
    if not res:
        print("   " + warn("애널리스트 데이터 없음 (목표가 미수집)"))
    else:
        for t in TEST_TICKERS:
            a = res.get(t, {})
            if a and a.get("target_mean"):
                print("   " + ok(f"{t:6s} 목표가={a.get('target_mean')} 업사이드={a.get('upside_pct')}"))
            else:
                print("   " + warn(f"{t:6s} 목표가 없음"))
except Exception as e:
    print("   " + bad(f"analyst 모듈 예외: {str(e)[:80]}"))

# ──────────────────────────────────────────────────────────────
# 종합 진단
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("종합 진단")
print("=" * 64)
n_price = len(price_ok)
n_info  = len(info_ok)
total   = len(TEST_TICKERS)

if n_price == 0:
    print(bad("가격 데이터부터 전부 실패 → 분석 자체가 안 됨 (네트워크/yfinance 문제)"))
    print("   해결: pip install -U yfinance, VPN/방화벽 확인")
elif n_price > 0 and n_info == 0:
    print(warn("가격은 되는데 info(재무/밸류에이션)만 전부 실패"))
    print("   → '분석 자체'는 되지만 '재무 데이터만' 안 들어오는 상황")
    print("   원인 후보:")
    print("     1. yfinance가 .info를 빈 dict로 반환 (버전 이슈가 가장 흔함)")
    print("     2. Yahoo 레이트리밋(429) — 잠시 후 재시도")
    print("   해결: pip install -U yfinance  (또는 특정 버전 고정)")
elif n_info < total:
    print(warn(f"일부만 재무 수집됨 ({n_info}/{total}) — 부분적 레이트리밋 가능성"))
    print("   → 시간을 두고 재실행하거나 종목 수를 줄여보세요")
else:
    print(ok("가격·재무·메타 모두 정상 → 데이터 수집은 문제없음"))
    print("   화면에서 비어 보였다면 캐시 문제일 수 있음 (점수만 재계산/완전 초기화)")

print()
