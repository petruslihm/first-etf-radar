"""
screener/fundamental.py

2단계 재무/실적 분석 모듈.
1단계(차트/수급)에서 선정된 Top N 종목에만 실행.

미국: yfinance info (PER, 성장률, 마진, ROE 등)
한국: FnGuide (매출/영업이익/순이익 분기·연간, 성장률)

Fundamental Score (0~100):
  매출 성장률 YoY        20점
  영업이익 성장률 YoY    20점
  EPS/순이익 성장률 YoY  20점
  성장 가속 여부         15점  (전분기 대비 성장률 개선)
  영업이익률 개선        10점
  밸류에이션 (PEG/PER)   10점  (낮을수록 가점)
  재무 안정성            5점   (부채비율 낮을수록)
"""

import logging
import math
import time
import random
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 재무 캐시 (per-ticker, 데이터 기준일로 키잉) ─────────────────────
# 재무는 야후의 무거운 .info 호출이라 rate limit에 가장 잘 걸린다.
# 성공한 종목은 캐시에 저장 → 재실행 시 야후를 다시 안 두드림(차단 회피).
# 실패(rate limit)한 종목은 캐시 안 함 → 다음 실행에서만 재시도.
def _fund_cache_path(ticker: str, market: str):
    from screener.paths import CACHE_DIR, data_date_key
    return CACHE_DIR / f"fund_{market}_{ticker}_{data_date_key()}.pkl"


def _fund_cache_get(ticker: str, market: str):
    p = _fund_cache_path(ticker, market)
    if p.exists():
        try:
            return pd.read_pickle(p)
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _fund_cache_put(ticker: str, market: str, data: dict):
    if not isinstance(data, dict) or not data.get("data_available"):
        return  # 실패/빈 데이터는 캐시하지 않음(다음에 재시도)
    try:
        pd.to_pickle(data, _fund_cache_path(ticker, market))
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════
# 미국 재무 데이터 (yfinance)
# ════════════════════════════════════════════════════════════════════

def fetch_us_fundamental(ticker: str) -> dict:
    """
    yfinance info + quarterly_income_stmt 로 재무 지표 수집.
    반환: {
      rev_growth, op_growth, eps_growth,  # YoY %
      rev_accel, margin_trend,            # 가속 여부, 마진 개선
      trailing_pe, forward_pe, peg,       # 밸류에이션
      profit_margin, op_margin, roe,
      debt_to_eq, fund_score, fund_grade
    }
    """
    result = _empty_fund()
    try:
        import yfinance as yf
        obj = yf.Ticker(ticker)
        try:
            raw_info = obj.info
            # yfinance 1.3.0에서 info가 None을 반환하는 경우 처리
            info = raw_info if isinstance(raw_info, dict) else {}
            if len(info) == 0:
                # fast_info로 최소한의 데이터 시도
                try:
                    fi = obj.fast_info
                    info = {
                        "currentPrice":   getattr(fi, "last_price", None),
                        "marketCap":      getattr(fi, "market_cap", None),
                        "fiftyTwoWeekHigh": getattr(fi, "year_high", None),
                        "fiftyTwoWeekLow":  getattr(fi, "year_low",  None),
                    }
                except Exception:
                    info = {}
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err or "too many" in err:
                raise  # 배치에서 재시도
            info = {}

        # ── info에서 직접 오는 값 ──────────────────────────────────
        rev_growth  = _pct(info.get("revenueGrowth"))        # YoY 매출 성장률
        eps_growth  = _pct(info.get("earningsQuarterlyGrowth"))  # 분기 EPS YoY
        op_margin   = _pct(info.get("operatingMargins"))
        profit_margin = _pct(info.get("profitMargins"))
        roe         = _pct(info.get("returnOnEquity"))
        trailing_pe = info.get("trailingPE")
        forward_pe  = info.get("forwardPE")
        peg         = info.get("pegRatio")
        debt_to_eq  = info.get("debtToEquity")
        trailing_eps = info.get("trailingEps")
        forward_eps  = info.get("forwardEps")

        # ── 분기 손익계산서에서 영업이익 성장률 계산 ──────────────
        op_growth  = float("nan")
        rev_accel  = False
        margin_trend = float("nan")

        try:
            q = obj.quarterly_income_stmt
            if q is not None and not q.empty and q.shape[1] >= 4:
                # 행 이름에서 영업이익, 매출 찾기
                rev_row = None
                op_row  = None
                for idx in q.index:
                    s = str(idx).lower()
                    if "total revenue" in s or "revenue" in s:
                        rev_row = idx
                    if "operating income" in s or "ebit" in s:
                        op_row = idx

                if op_row is not None:
                    ops = q.loc[op_row].dropna()
                    if len(ops) >= 5:
                        # 최신 분기 vs 1년 전 분기
                        op_now  = float(ops.iloc[0])
                        op_yr   = float(ops.iloc[4])
                        op_prev = float(ops.iloc[1]) if len(ops) > 1 else float("nan")
                        if op_yr != 0 and math.isfinite(op_yr):
                            op_growth = (op_now - op_yr) / abs(op_yr) * 100
                        # 성장 가속: 최신 분기 성장률 > 직전 분기 성장률
                        if math.isfinite(op_prev) and op_yr != 0:
                            prev_yr = float(ops.iloc[5]) if len(ops) > 5 else float("nan")
                            if math.isfinite(prev_yr) and prev_yr != 0:
                                growth_now  = (op_now  - op_yr)  / abs(op_yr)
                                growth_prev = (op_prev - prev_yr) / abs(prev_yr)
                                rev_accel = growth_now > growth_prev

                # 마진 트렌드 (최신 - 4분기 전 마진 차이)
                if rev_row is not None and op_row is not None:
                    revs = q.loc[rev_row].dropna()
                    ops2 = q.loc[op_row].dropna()
                    if len(revs) >= 5 and len(ops2) >= 5:
                        margin_now = float(ops2.iloc[0]) / float(revs.iloc[0]) * 100 \
                            if float(revs.iloc[0]) != 0 else float("nan")
                        margin_yr  = float(ops2.iloc[4]) / float(revs.iloc[4]) * 100 \
                            if float(revs.iloc[4]) != 0 else float("nan")
                        if math.isfinite(margin_now) and math.isfinite(margin_yr):
                            margin_trend = margin_now - margin_yr

        except Exception as e:
            logger.debug(f"분기 손익 파싱({ticker}): {e}")

        result.update({
            "rev_growth":    rev_growth,
            "op_growth":     op_growth,
            "eps_growth":    eps_growth,
            "rev_accel":     rev_accel,
            "margin_trend":  margin_trend,
            "trailing_pe":   trailing_pe,
            "forward_pe":    forward_pe,
            "peg":           peg,
            "profit_margin": profit_margin,
            "op_margin":     op_margin,
            "roe":           roe,
            "debt_to_eq":    debt_to_eq,
            "trailing_eps":  trailing_eps,
            "forward_eps":   forward_eps,
            "data_available": True,   # 수집 성공
        })

    except Exception as e:
        logger.debug(f"US 재무({ticker}): {e}")

    # 핵심 지표 중 하나라도 있으면 수집 성공으로 표시
    has_data = any(
        math.isfinite(float(result.get(k) or float("nan")))
        for k in ["rev_growth","eps_growth","op_margin","roe","forward_pe"]
    )
    result["data_available"] = has_data
    result["fund_score"] = _calc_us_fund_score(result) if has_data else None
    result["fund_grade"] = _grade(result["fund_score"]) if has_data and result["fund_score"] is not None else ""
    return result


# ════════════════════════════════════════════════════════════════════
# 한국 재무 데이터 (FnGuide)
# ════════════════════════════════════════════════════════════════════

def fetch_kr_fundamental(code: str) -> dict:
    """
    FnGuide에서 연간/분기 실적 파싱.
    테이블0: 연간 (매출, 영업이익, 순이익, EPS, PER, ROE ...)
    테이블1: 분기 (전년동기비 포함)
    """
    result = _empty_fund()
    try:
        import requests
        from io import StringIO

        s = requests.Session()
        s.headers.update({
            "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer":       "http://comp.fnguide.com/",
        })
        r = s.get(
            f"http://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp"
            f"?pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=D&NewMenuID=103&stkGb=701",
            timeout=12
        )
        if r.status_code != 200:
            return result

        tbls = pd.read_html(StringIO(r.text), encoding="cp949")
        if not tbls:
            return result

        # ── 연간 테이블 파싱 ──────────────────────────────────────
        ann = tbls[0] if len(tbls) > 0 else None
        qtr = tbls[1] if len(tbls) > 1 else None

        def _row(df, keywords):
            """키워드 포함 행 반환."""
            if df is None: return None
            first_col = df.columns[0]
            for kw in keywords:
                mask = df[first_col].astype(str).str.contains(kw, na=False)
                if mask.any():
                    return df[mask].iloc[0]
            return None

        def _num(val):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return float("nan")
            try:
                return float(str(val).replace(",", "").replace("%", ""))
            except Exception:
                return float("nan")

        # 연간 성장률 (최근 2개년)
        rev_growth = op_growth = eps_growth = float("nan")
        if ann is not None and ann.shape[1] >= 3:
            rev_row = _row(ann, ["매출액"])
            op_row  = _row(ann, ["영업이익"])
            eps_row = _row(ann, ["EPS"])

            # 최근 연도 컬럼 (마지막 2개)
            num_cols = [c for c in ann.columns[1:] if str(c) not in ["전년동기","전년동기(%)"]]
            if len(num_cols) >= 2:
                c_now = num_cols[-1]
                c_prev= num_cols[-2]
                if rev_row is not None:
                    v_now, v_prev = _num(rev_row.get(c_now)), _num(rev_row.get(c_prev))
                    if math.isfinite(v_now) and math.isfinite(v_prev) and v_prev != 0:
                        rev_growth = (v_now - v_prev) / abs(v_prev) * 100
                if op_row is not None:
                    v_now, v_prev = _num(op_row.get(c_now)), _num(op_row.get(c_prev))
                    if math.isfinite(v_now) and math.isfinite(v_prev) and v_prev != 0:
                        op_growth = (v_now - v_prev) / abs(v_prev) * 100
                if eps_row is not None:
                    v_now, v_prev = _num(eps_row.get(c_now)), _num(eps_row.get(c_prev))
                    if math.isfinite(v_now) and math.isfinite(v_prev) and v_prev != 0:
                        eps_growth = (v_now - v_prev) / abs(v_prev) * 100

        # 분기 전년동기비 (가장 최근 분기)
        rev_accel = False
        margin_trend = float("nan")
        if qtr is not None and "전년동기(%)" in qtr.columns:
            rev_row_q = _row(qtr, ["매출액"])
            op_row_q  = _row(qtr, ["영업이익"])
            if rev_row_q is not None:
                rev_yoy = _num(rev_row_q.get("전년동기(%)"))
            if op_row_q is not None:
                # 가장 최근 2개 분기 영업이익률 비교
                num_q = [c for c in qtr.columns[1:] if str(c) not in ["전년동기","전년동기(%)"]]
                if len(num_q) >= 2:
                    rev_r = _row(qtr, ["매출액"])
                    if rev_r is not None and op_row_q is not None:
                        r_now  = _num(rev_r.get(num_q[-1]))
                        r_prev = _num(rev_r.get(num_q[-2]))
                        o_now  = _num(op_row_q.get(num_q[-1]))
                        o_prev = _num(op_row_q.get(num_q[-2]))
                        if r_now and r_prev and r_now != 0 and r_prev != 0:
                            m_now  = o_now  / r_now  * 100
                            m_prev = o_prev / r_prev * 100
                            if math.isfinite(m_now) and math.isfinite(m_prev):
                                margin_trend = m_now - m_prev
                                rev_accel    = margin_trend > 0

        # PER / PBR (연간 테이블에서 찾기)
        trailing_pe = forward_pe = float("nan")
        if ann is not None:
            per_row = _row(ann, ["PER"])
            pbr_row = _row(ann, ["PBR"])
            num_cols2 = [c for c in ann.columns[1:] if str(c) not in ["전년동기","전년동기(%)"]]
            if per_row is not None and num_cols2:
                trailing_pe = _num(per_row.get(num_cols2[-1]))
            # ROE
            roe_row = _row(ann, ["ROE"])
            roe = _num(roe_row.get(num_cols2[-1])) if roe_row is not None and num_cols2 else float("nan")

        result.update({
            "rev_growth":    rev_growth,
            "op_growth":     op_growth,
            "eps_growth":    eps_growth,
            "rev_accel":     rev_accel,
            "margin_trend":  margin_trend,
            "trailing_pe":   trailing_pe if math.isfinite(trailing_pe) else float("nan"),
            "forward_pe":    float("nan"),
            "peg":           float("nan"),
            "roe":           roe if "roe" in dir() else float("nan"),
        })

    except Exception as e:
        logger.debug(f"KR 재무({code}): {e}")

    has_data = any(
        math.isfinite(float(result.get(k) or float("nan")))
        for k in ["rev_growth","op_growth","eps_growth","roe"]
    )
    result["data_available"] = has_data
    result["fund_score"] = _calc_kr_fund_score(result) if has_data else None
    result["fund_grade"] = _grade(result["fund_score"]) if has_data and result["fund_score"] is not None else ""
    return result


# ════════════════════════════════════════════════════════════════════
# Fundamental Score 계산
# ════════════════════════════════════════════════════════════════════

def _empty_fund() -> dict:
    return {
        "rev_growth": float("nan"), "op_growth": float("nan"),
        "eps_growth": float("nan"), "rev_accel": False,
        "margin_trend": float("nan"), "trailing_pe": None,
        "forward_pe": None, "peg": None, "profit_margin": float("nan"),
        "op_margin": float("nan"), "roe": float("nan"),
        "debt_to_eq": None, "trailing_eps": None, "forward_eps": None,
        "fund_score":  None,   # None = 수집 실패 (50점 착시 방지)
        "fund_grade":  "",     # 빈 문자열 = 데이터 없음
        "data_available": False,
    }


def _pct(v) -> float:
    if v is None: return float("nan")
    try:
        v = float(v)
        return v * 100 if abs(v) < 10 else v  # 0.73 → 73%
    except Exception:
        return float("nan")


def _clamp(v, lo, hi, neutral=0):
    if not math.isfinite(float(v if v is not None else float("nan"))):
        return neutral
    return max(lo, min(hi, float(v)))


def _calc_us_fund_score(d: dict) -> float:
    score = 50.0

    # 1. 매출 성장률 YoY (0~20점)
    rev = d.get("rev_growth", float("nan"))
    if math.isfinite(float(rev if rev else float("nan"))):
        rev = float(rev)
        if rev >= 50:   score += 20
        elif rev >= 25: score += 15
        elif rev >= 15: score += 10
        elif rev >= 5:  score += 5
        elif rev < 0:   score -= 10

    # 2. EPS 분기 성장률 YoY (0~20점)
    eps = d.get("eps_growth", float("nan"))
    if math.isfinite(float(eps if eps else float("nan"))):
        eps = float(eps)
        if eps >= 80:   score += 20
        elif eps >= 40: score += 15
        elif eps >= 20: score += 10
        elif eps >= 5:  score += 5
        elif eps < -10: score -= 10

    # 3. 영업이익 성장률 (0~20점)
    op = d.get("op_growth", float("nan"))
    if math.isfinite(float(op if op else float("nan"))):
        op = float(op)
        if op >= 60:    score += 20
        elif op >= 30:  score += 14
        elif op >= 15:  score += 8
        elif op >= 0:   score += 3
        elif op < -20:  score -= 10

    # 4. 성장 가속 (0~15점)
    if d.get("rev_accel"):
        score += 15

    # 5. 마진 개선 (0~10점)
    mt = d.get("margin_trend", float("nan"))
    if math.isfinite(float(mt if mt else float("nan"))):
        mt = float(mt)
        if mt >= 5:    score += 10
        elif mt >= 2:  score += 6
        elif mt >= 0:  score += 3
        elif mt < -5:  score -= 8

    # 6. 밸류에이션 (PEG 기준, 없으면 Forward PE)
    peg = d.get("peg")
    fpe = d.get("forward_pe")
    if peg is not None and math.isfinite(float(peg)):
        peg = float(peg)
        if peg < 0:    pass           # 적자 또는 이상값
        elif peg < 1:  score += 10   # 매력적
        elif peg < 2:  score += 5
        elif peg > 4:  score -= 8    # 과대평가
    elif fpe is not None and math.isfinite(float(fpe)):
        fpe = float(fpe)
        if fpe < 15:   score += 8
        elif fpe < 25: score += 4
        elif fpe > 50: score -= 8

    # 7. 재무 안정성 (부채비율)
    de = d.get("debt_to_eq")
    if de is not None and math.isfinite(float(de)):
        de = float(de)
        if de < 50:    score += 5
        elif de < 100: score += 2
        elif de > 300: score -= 5

    return round(max(0, min(100, score)), 1)


def _calc_kr_fund_score(d: dict) -> float:
    score = 50.0

    rev = d.get("rev_growth", float("nan"))
    op  = d.get("op_growth",  float("nan"))
    eps = d.get("eps_growth", float("nan"))

    def safe(v):
        if v is None: return float("nan")
        try: return float(v)
        except: return float("nan")

    rev, op, eps = safe(rev), safe(op), safe(eps)

    if math.isfinite(rev):
        if rev >= 30:   score += 18
        elif rev >= 15: score += 12
        elif rev >= 5:  score += 6
        elif rev < 0:   score -= 10

    if math.isfinite(op):
        if op >= 50:    score += 20
        elif op >= 25:  score += 14
        elif op >= 10:  score += 8
        elif op >= 0:   score += 3
        elif op < -30:  score -= 12

    if math.isfinite(eps):
        if eps >= 50:   score += 15
        elif eps >= 20: score += 10
        elif eps >= 5:  score += 5
        elif eps < -20: score -= 8

    if d.get("rev_accel"):
        score += 15

    mt = safe(d.get("margin_trend"))
    if math.isfinite(mt):
        if mt >= 3:   score += 8
        elif mt >= 1: score += 4
        elif mt < -3: score -= 6

    pe = safe(d.get("trailing_pe"))
    if math.isfinite(pe) and pe > 0:
        if pe < 10:    score += 8
        elif pe < 20:  score += 4
        elif pe > 50:  score -= 6

    return round(max(0, min(100, score)), 1)


def _grade(score: float) -> str:
    if score >= 80: return "A+"
    if score >= 70: return "A"
    if score >= 60: return "B+"
    if score >= 50: return "B"
    if score >= 40: return "C"
    return "D"


# ════════════════════════════════════════════════════════════════════
# 배치 수집 (Top N에만 적용)
# ════════════════════════════════════════════════════════════════════

def fetch_fundamentals_batch(
    tickers:    list[str],
    market:     str = "US",
    progress_cb = None,
) -> dict[str, dict]:
    """
    Top N 종목의 재무 데이터 일괄 수집.
    Rate limit 방지:
    - 미국: 종목당 1.2초 간격, rate limit 시 15초 대기 후 재시도 1회
    - 한국: 종목당 0.5초 간격
    """
    results = {}
    total   = len(tickers)
    n_cached = 0

    for i, t in enumerate(tickers):
        if progress_cb:
            progress_cb(i + 1, total, t)
        # 1) 캐시 우선 — 야후 호출 자체를 건너뛴다(rate limit 회피)
        cached = _fund_cache_get(t, market)
        if cached is not None:
            results[t] = cached; n_cached += 1
            continue
        try:
            if market == "US":
                results[t] = fetch_us_fundamental(t)
                _fund_cache_put(t, market, results[t])
                time.sleep(1.2 + random.uniform(0, 0.6))   # 간격 + 지터
            else:
                results[t] = fetch_kr_fundamental(t)
                _fund_cache_put(t, market, results[t])
                time.sleep(0.5)
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "too many" in err_str or "429" in err_str:
                logger.warning(f"Rate limit 감지 ({t}), 20초 대기 후 재시도...")
                time.sleep(20)
                try:
                    results[t] = fetch_us_fundamental(t) if market=="US" else fetch_kr_fundamental(t)
                    _fund_cache_put(t, market, results[t])
                except Exception as e2:
                    logger.debug(f"재무 재시도 실패({t}): {e2}")
                    results[t] = _empty_fund()
            else:
                logger.debug(f"재무({t}): {e}")
                results[t] = _empty_fund()

    n_success = sum(1 for v in results.values() if v.get("data_available",False))
    n_fail    = len(results) - n_success
    if n_cached:
        logger.info(f"재무 캐시 재사용 {n_cached}개 (야후 호출 생략)")
    if n_fail > 0 and n_success == 0:
        logger.warning(
            f"재무 수집 전체 실패 ({n_fail}개) — Yahoo Finance 네트워크 차단 가능성. "
            "로컬 PC에서 실행 시 정상 수집됩니다."
        )
    elif n_fail > 0:
        logger.info(f"재무 수집: 성공 {n_success}개 / 실패 {n_fail}개")
    return results
