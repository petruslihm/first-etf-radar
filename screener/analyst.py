"""
screener/analyst.py

애널리스트 데이터 + 실적 이벤트 + 컨센서스 상향 감지.
yfinance로 무료 수집.

수집 항목:
  - 애널리스트 목표가 (평균/최고/최저)
  - 컨센서스 상향/하향 여부 (최근 EPS 추정치 변화)
  - 다음 실적 발표일
  - 실적 서프라이즈 히스토리 (최근 4분기)
  - 매수/중립/매도 비율

Analyst Score (0~100):
  목표가 상향 여부     20점
  컨센서스 EPS 상향    25점
  실적 서프라이즈 연속 20점
  매수 비율 높음       15점
  목표가 대비 업사이드 20점
"""

import logging
import math
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 미국 — yfinance 애널리스트 데이터
# ════════════════════════════════════════════════════════════════════

def fetch_us_analyst(ticker: str) -> dict:
    """
    yfinance에서 애널리스트 목표가, 컨센서스, 실적 히스토리 수집.
    """
    result = _empty_analyst()
    try:
        import yfinance as yf
        obj  = yf.Ticker(ticker)
        info = obj.info or {}

        cur_price    = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        target_mean  = info.get("targetMeanPrice")
        target_high  = info.get("targetHighPrice")
        target_low   = info.get("targetLowPrice")
        num_analysts = info.get("numberOfAnalystOpinions", 0)
        recommend    = info.get("recommendationKey", "")  # strong_buy, buy, hold, sell
        recommend_mean = info.get("recommendationMean", 3.0)  # 1=Strong Buy, 5=Strong Sell

        # 업사이드 계산
        upside = float("nan")
        if cur_price and target_mean and cur_price > 0:
            upside = (target_mean / cur_price - 1) * 100

        # 매수 비율
        buy_pct = float("nan")
        try:
            recs = obj.recommendations
            if recs is not None and not recs.empty:
                # 최근 1개월 추천
                recent = recs.tail(1)
                if not recent.empty:
                    cols = [c for c in recent.columns if c not in ["period","firm"]]
                    row  = recent.iloc[0]
                    buy_cnt  = sum(row.get(c, 0) for c in cols if "buy" in str(c).lower() or "strong" in str(c).lower())
                    sell_cnt = sum(row.get(c, 0) for c in cols if "sell" in str(c).lower() or "under" in str(c).lower())
                    hold_cnt = sum(row.get(c, 0) for c in cols if "hold" in str(c).lower() or "neutral" in str(c).lower())
                    total = buy_cnt + sell_cnt + hold_cnt
                    if total > 0:
                        buy_pct = buy_cnt / total * 100
        except Exception:
            pass

        # EPS 컨센서스 트렌드 (상향/하향)
        eps_trend    = float("nan")
        eps_revision = "neutral"
        try:
            trend = obj.earnings_trend
            if trend is not None and not trend.empty:
                # 현재 분기 EPS 추정치 변화
                cur_q = trend[trend.index == "0q"] if hasattr(trend, "index") else None
                if cur_q is None or cur_q.empty:
                    cur_q = trend.head(1)
                if not cur_q.empty:
                    eps_now  = cur_q.get("earningsEstimate.avg", pd.Series([float("nan")])).iloc[0]
                    eps_7d   = cur_q.get("earningsEstimate.7daysAgo", pd.Series([float("nan")])).iloc[0]
                    eps_30d  = cur_q.get("earningsEstimate.30daysAgo", pd.Series([float("nan")])).iloc[0]
                    if math.isfinite(float(eps_now or float("nan"))) and \
                       math.isfinite(float(eps_30d or float("nan"))) and float(eps_30d) != 0:
                        eps_trend = (float(eps_now) / float(eps_30d) - 1) * 100
                        if eps_trend >= 2:    eps_revision = "up"
                        elif eps_trend <= -2: eps_revision = "down"
        except Exception:
            pass

        # 실적 서프라이즈 히스토리
        beat_count = 0; miss_count = 0; surprise_avg = float("nan")
        try:
            hist = obj.earnings_history
            if hist is not None and not hist.empty:
                recent4 = hist.tail(4)
                surprises = []
                for _, row in recent4.iterrows():
                    actual = row.get("epsActual", float("nan"))
                    est    = row.get("epsEstimate", float("nan"))
                    if math.isfinite(float(actual or float("nan"))) and \
                       math.isfinite(float(est or float("nan"))) and float(est) != 0:
                        pct = (float(actual) - float(est)) / abs(float(est)) * 100
                        surprises.append(pct)
                        if pct > 0: beat_count += 1
                        else: miss_count += 1
                if surprises:
                    surprise_avg = float(np.mean(surprises))
        except Exception:
            pass

        # 다음 실적 발표일
        next_earnings = None
        try:
            cal = obj.calendar
            if cal is not None and not cal.empty:
                ec = [c for c in cal.columns if "earnings" in str(c).lower()]
                if ec:
                    next_earnings = str(cal[ec[0]].iloc[0])[:10]
        except Exception:
            pass

        # 실적 발표까지 남은 일수
        days_to_earnings = float("nan")
        if next_earnings:
            try:
                from datetime import date
                ned = pd.to_datetime(next_earnings).date()
                days_to_earnings = (ned - date.today()).days
            except Exception:
                pass

        result.update({
            "target_mean":       target_mean,
            "target_high":       target_high,
            "target_low":        target_low,
            "upside_pct":        round(upside, 1) if math.isfinite(upside) else float("nan"),
            "num_analysts":      num_analysts,
            "recommend_key":     recommend,
            "recommend_mean":    recommend_mean,
            "buy_pct":           round(buy_pct, 1) if math.isfinite(buy_pct) else float("nan"),
            "eps_trend":         round(eps_trend, 2) if math.isfinite(eps_trend) else float("nan"),
            "eps_revision":      eps_revision,
            "beat_count":        beat_count,
            "miss_count":        miss_count,
            "surprise_avg":      round(surprise_avg, 2) if math.isfinite(surprise_avg) else float("nan"),
            "next_earnings":     next_earnings,
            "days_to_earnings":  int(days_to_earnings) if math.isfinite(days_to_earnings) else None,
        })

    except Exception as e:
        logger.debug(f"애널리스트({ticker}): {e}")

    result["analyst_score"] = _calc_analyst_score(result)
    return result


# ════════════════════════════════════════════════════════════════════
# Analyst Score 계산
# ════════════════════════════════════════════════════════════════════

def _calc_analyst_score(d: dict) -> float:
    score = 50.0

    # 1. 목표가 업사이드 (0~20점)
    upside = d.get("upside_pct", float("nan"))
    if math.isfinite(float(upside if upside else float("nan"))):
        upside = float(upside)
        if upside >= 30:    score += 20
        elif upside >= 20:  score += 14
        elif upside >= 10:  score += 8
        elif upside >= 0:   score += 3
        elif upside < -10:  score -= 10  # 이미 목표가 초과

    # 2. EPS 컨센서스 상향 (0~25점)
    eps_rev = d.get("eps_revision", "neutral")
    eps_trend = d.get("eps_trend", float("nan"))
    if eps_rev == "up":
        trend_v = float(eps_trend if eps_trend else 0)
        if trend_v >= 10:   score += 25
        elif trend_v >= 5:  score += 18
        else:               score += 10
    elif eps_rev == "down":
        score -= 15

    # 3. 실적 서프라이즈 연속 (0~20점)
    beat = d.get("beat_count", 0); miss = d.get("miss_count", 0)
    surp = d.get("surprise_avg", float("nan"))
    if beat >= 4:           score += 20
    elif beat >= 3:         score += 14
    elif beat >= 2:         score += 8
    if miss >= 3:           score -= 12

    if math.isfinite(float(surp if surp else float("nan"))):
        surp = float(surp)
        if surp >= 10:      score += 8
        elif surp >= 5:     score += 4
        elif surp < -5:     score -= 8

    # 4. 매수 비율 (0~15점)
    buy_pct = d.get("buy_pct", float("nan"))
    if math.isfinite(float(buy_pct if buy_pct else float("nan"))):
        buy_pct = float(buy_pct)
        if buy_pct >= 80:   score += 15
        elif buy_pct >= 65: score += 10
        elif buy_pct >= 50: score += 5
        elif buy_pct < 30:  score -= 10

    # 5. 실적 발표 임박 + 컨센서스 상향 (이벤트 가속)
    dte = d.get("days_to_earnings")
    if dte is not None and 0 <= dte <= 14 and eps_rev == "up":
        score += 10   # 실적 발표 2주 이내 + 컨센서스 상향

    return round(max(0, min(100, score)), 1)


def _empty_analyst() -> dict:
    return {
        "target_mean": None, "target_high": None, "target_low": None,
        "upside_pct": float("nan"), "num_analysts": 0,
        "recommend_key": "", "recommend_mean": 3.0, "buy_pct": float("nan"),
        "eps_trend": float("nan"), "eps_revision": "neutral",
        "beat_count": 0, "miss_count": 0, "surprise_avg": float("nan"),
        "next_earnings": None, "days_to_earnings": None,
        "analyst_score": 50.0,
    }


# ════════════════════════════════════════════════════════════════════
# 배치 수집
# ════════════════════════════════════════════════════════════════════

def fetch_analyst_batch(
    tickers: list[str],
    progress_cb=None,
) -> dict[str, dict]:
    results = {}
    for i, t in enumerate(tickers):
        if progress_cb:
            progress_cb(i+1, len(tickers), t)
        results[t] = fetch_us_analyst(t)
        time.sleep(0.2)
    return results


# ════════════════════════════════════════════════════════════════════
# 실적 이벤트 캘린더 (앞으로 4주 내 실적 예정 종목)
# ════════════════════════════════════════════════════════════════════

def get_earnings_calendar(analyst_data: dict) -> list[dict]:
    """
    수집된 애널리스트 데이터에서 실적 발표 예정 종목 정리.
    Returns: [{"ticker", "days_to_earnings", "next_earnings", "eps_revision", "analyst_score"}]
    """
    events = []
    for ticker, d in analyst_data.items():
        dte = d.get("days_to_earnings")
        if dte is not None and 0 <= dte <= 28:
            events.append({
                "ticker":         ticker,
                "days_to_earnings": dte,
                "next_earnings":  d.get("next_earnings",""),
                "eps_revision":   d.get("eps_revision","neutral"),
                "analyst_score":  d.get("analyst_score", 50),
                "upside_pct":     d.get("upside_pct", float("nan")),
                "buy_pct":        d.get("buy_pct", float("nan")),
            })
    events.sort(key=lambda x: x["days_to_earnings"])
    return events
