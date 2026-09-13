"""
screener/events_us.py
US SEC EDGAR 기반 이벤트/촉매 레이어 v1

on-demand US 분석에서만 사용한다. 전체 스크리너 점수·backtest에는 영향 없음.
SEC EDGAR는 공개 API (별도 API 키 불필요). User-Agent 헤더만 필요.
SEC_USER_AGENT 환경변수로 재정의 가능; 없으면 기본값 사용.
네트워크/CIK 실패 시 event_score=None 반환 — 기존 점수 변경하지 않는다.
"""
import logging
import os
from datetime import date, timedelta

import requests

logger = logging.getLogger(__name__)

SEC_USER_AGENT_ENV = "SEC_USER_AGENT"
_DEFAULT_USER_AGENT  = "etf-radar-ondemand research@example.com"
_SEC_BASE_URL        = "https://data.sec.gov"
_SEC_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
_REQUEST_TIMEOUT     = 15  # 초

# ── 이벤트 분류 키워드 / 폼 타입 ─────────────────────────────────────

# 이 폼 타입은 희석/공모 → 항상 negative
_OFFERING_FORMS = frozenset({
    "S-1", "S-1/A", "S-3", "S-3/A",
    "424B1", "424B2", "424B3", "424B4", "424B5",
    "F-1", "F-1/A", "F-3", "F-3/A",
})

# 오너십·행정용 → 항상 neutral
_ADMIN_FORMS = frozenset({
    "3", "3/A", "4", "4/A", "5", "5/A",
    "SC 13G", "SC 13G/A", "SC 13D", "SC 13D/A",
    "DEF 14A", "PRE 14A", "DEFA14A",
})

# description 대문자 포함 여부 — 부정 우선 평가
_NEGATIVE_KW = (
    "OFFERING",
    "DILUTION", "DILUTIVE",
    "DELISTING", "DELIST",
    "BANKRUPTCY", "CHAPTER 11", "CHAPTER 7",
    "RESIGNATION",
    "INVESTIGATION", "SEC INQUIRY", "SUBPOENA",
    "LAWSUIT", "LITIGATION",
    "IMPAIRMENT", "WRITE-DOWN", "WRITE DOWN",
    "GOING CONCERN",
    "RESTATEMENT", "RESTATE",
    "NT 10-K", "NT 10-Q",          # 지연 공시 알림
    "CONVERTIBLE NOTE", "CONVERTIBLE DEBT",
    "DEBT OFFERING", "SENIOR NOTE",
)

# description 대문자 포함 여부 — 긍정
_POSITIVE_KW = (
    "EARNINGS", "REVENUE", "EPS",
    "REPURCHASE", "BUYBACK", "BUY BACK", "SHARE REPURCHASE",
    "DIVIDEND",
    "ACQUISITION", "ACQUIRES", "MERGER", "DEFINITIVE AGREEMENT",
    "AGREEMENT", "CONTRACT", "PARTNERSHIP",
    "GUIDANCE", "OUTLOOK", "FORECAST",
    "APPROVAL", "FDA", "CLEARED", "AUTHORIZED",
    "PRODUCT LAUNCH",
    "BEATS", "BEAT ESTIMATES",
    "RECORD REVENUE", "RECORD EARNINGS",
    "GROWTH",
)

# 10-Q / 10-K 는 자체로는 neutral 이지만 scoring 에서 미세 +5
_PERIODIC_FORMS = frozenset({"10-Q", "10-Q/A", "10-K", "10-K/A"})


# ─── 공개 API ─────────────────────────────────────────────────────────


def classify_edgar_event(form: str, description: str) -> str:
    """
    SEC EDGAR 공시를 positive / negative / neutral 로 분류한다.
    부정 신호가 긍정 신호보다 우선한다.
    """
    form = str(form or "").strip().upper()
    desc = str(description or "").strip().upper()

    # 1. 공모/희석 폼 → always negative
    if form in _OFFERING_FORMS:
        return "negative"

    # 2. 행정·오너십 폼 → always neutral
    if form in _ADMIN_FORMS:
        return "neutral"

    # 3. description 부정 키워드
    for kw in _NEGATIVE_KW:
        if kw in desc:
            return "negative"

    # 4. description 긍정 키워드
    for kw in _POSITIVE_KW:
        if kw in desc:
            return "positive"

    # 5. 8-K without signal → neutral
    # 10-Q/10-K without signal → neutral (score 함수에서 +5)
    return "neutral"


def score_edgar_events(classified: "list[dict]") -> int:
    """
    분류된 이벤트 목록으로 0~100 정수 점수를 계산한다.

    base = 50
    positive 8-K / material:  +12 each, max +25 총합 상한
    10-Q / 10-K (정기 공시):   +5 총합 (부정 없을 때만)
    negative event:            -18 each, max -35 총합 하한
    clamp 0~100
    """
    base = 50
    pos_delta = 0
    neg_delta = 0
    has_periodic = False
    has_negative = False

    for ev in classified:
        cls  = ev.get("classification", "neutral")
        form = str(ev.get("form", "")).strip().upper()

        if cls == "positive":
            pos_delta = min(pos_delta + 12, 25)
        elif cls == "negative":
            neg_delta = min(neg_delta + 18, 35)
            has_negative = True
        elif form in _PERIODIC_FORMS:
            has_periodic = True

    if has_periodic and not has_negative:
        pos_delta = min(pos_delta + 5, 25)

    score = base + pos_delta - neg_delta
    return max(0, min(100, score))


def _get_user_agent() -> str:
    return os.environ.get(SEC_USER_AGENT_ENV) or _DEFAULT_USER_AGENT


def _get_cik(ticker: str, user_agent: str) -> "str | None":
    """
    SEC company_tickers.json 에서 ticker → CIK 문자열 반환.
    실패/미발견 시 None.
    """
    try:
        resp = requests.get(
            _SEC_TICKERS_URL,
            headers={"User-Agent": user_agent},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        tk_upper = ticker.strip().upper()
        for entry in data.values():
            if str(entry.get("ticker", "")).upper() == tk_upper:
                return str(entry["cik_str"])
        return None
    except Exception as exc:
        logger.debug("CIK 조회 실패 (%s): %s", ticker, exc)
        return None


def _get_filings(cik: str, user_agent: str, days: int) -> "list[dict]":
    """
    EDGAR submissions JSON 에서 최근 `days` 일치 이내 공시 목록 반환.
    실패 시 빈 목록.
    """
    padded = str(cik).zfill(10)
    url    = f"{_SEC_BASE_URL}/submissions/CIK{padded}.json"
    resp   = requests.get(
        url, headers={"User-Agent": user_agent}, timeout=_REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    data   = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    dates  = recent.get("filingDate", [])
    descs  = recent.get("description", [])
    accnos = recent.get("accessionNumber", [])

    cutoff = (date.today() - timedelta(days=days)).isoformat()

    results = []
    for form, dt, desc, accno in zip(forms, dates, descs, accnos):
        if str(dt) < cutoff:
            continue
        results.append({
            "form":           form,
            "filingDate":     dt,
            "description":    desc or "",
            "accessionNumber": accno,
        })

    return results[:20]


def fetch_us_edgar_events(ticker: str, days: int = 30) -> dict:
    """
    SEC EDGAR 공시 기반 US 이벤트 촉매 정보를 반환한다.

    반환 dict:
      ticker, cik, events, event_score (0-100 or None),
      event_flags, positive_events, negative_events, neutral_events,
      error (None or str), source
    """
    base: dict = {
        "ticker":          ticker,
        "cik":             None,
        "events":          [],
        "event_score":     None,
        "event_flags":     [],
        "positive_events": [],
        "negative_events": [],
        "neutral_events":  [],
        "error":           None,
        "source":          "SEC EDGAR",
    }

    user_agent = _get_user_agent()

    # CIK 조회
    try:
        cik = _get_cik(ticker, user_agent)
    except Exception as exc:
        base["error"] = f"{ticker}: CIK 조회 중 오류 — {exc}"
        return base

    if not cik:
        base["error"] = (
            f"{ticker}: SEC EDGAR CIK를 찾을 수 없습니다. "
            "유효한 US 티커인지 확인하세요."
        )
        return base

    base["cik"] = cik

    # 공시 목록 조회
    try:
        raw_filings = _get_filings(cik, user_agent, days)
    except Exception as exc:
        base["error"] = f"{ticker}: EDGAR 공시 조회 중 오류 — {exc}"
        return base

    # 분류
    classified = []
    for f in raw_filings:
        cls = classify_edgar_event(f["form"], f["description"])
        classified.append({**f, "classification": cls})

    # 분류별 분류
    pos  = [e for e in classified if e["classification"] == "positive"]
    neg  = [e for e in classified if e["classification"] == "negative"]
    neu  = [e for e in classified if e["classification"] == "neutral"]

    # 플래그 — 최대 5개씩
    flags = []
    for e in pos[:3]:
        form = e["form"]
        desc = (e["description"] or form)[:40]
        flags.append(f"✅ {e['filingDate']} [{form}] {desc}")
    for e in neg[:3]:
        form = e["form"]
        desc = (e["description"] or form)[:40]
        flags.append(f"⚠ {e['filingDate']} [{form}] {desc}")

    score = score_edgar_events(classified)

    base["events"]          = classified[:10]
    base["event_score"]     = score
    base["event_flags"]     = flags
    base["positive_events"] = pos
    base["negative_events"] = neg
    base["neutral_events"]  = neu

    return base


# ── 실적 촉매 ─────────────────────────────────────────────────────────


def _score_earnings(eps_s: "float | None", rev_s: "float | None") -> "tuple[int, list[str]]":
    """
    EPS 서프라이즈(%)와 매출 서프라이즈(%)로 0~100 정수 점수 + flags 반환.
    데이터가 모두 None이면 (None, []) 반환.
    """
    if eps_s is None and rev_s is None:
        return None, []  # type: ignore[return-value]

    score = 50
    flags: list[str] = []

    if eps_s is not None:
        if eps_s > 10:
            score += 15
            flags.append(f"✅ EPS서프라이즈+{eps_s:.1f}%")
        elif eps_s > 3:
            score += 8
            flags.append(f"✅ EPS비트+{eps_s:.1f}%")
        elif eps_s < -5:
            score -= 12
            flags.append(f"⚠ EPS미스{eps_s:.1f}%")

    if rev_s is not None:
        if rev_s > 5:
            score += 8
            flags.append(f"✅ 매출비트+{rev_s:.1f}%")
        elif rev_s < -5:
            score -= 8
            flags.append(f"⚠ 매출미스{rev_s:.1f}%")

    # 두 항목 동시 beat 보너스
    if eps_s is not None and rev_s is not None and eps_s > 0 and rev_s > 0:
        score += 5
        flags.append("✅ EPS+매출동시비트")

    return max(0, min(100, score)), flags


def fetch_us_earnings_catalyst(ticker: str, item: "dict | None" = None) -> dict:
    """
    yfinance에서 최근 분기 실적 EPS 서프라이즈를 가져온다.
    item dict가 있으면 earnings_date를 재사용.
    non-fatal: 실패해도 기존 on-demand 결과에 영향 없음.
    validated_score / rr_score / catalyst_score 변경 없음.

    반환 dict:
      ticker, earnings_score (0-100 or None),
      earnings_flags, latest_earnings_date,
      eps_surprise_pct, revenue_surprise_pct,
      error, source
    """
    base: dict = {
        "ticker":               ticker,
        "earnings_score":       None,
        "earnings_flags":       [],
        "latest_earnings_date": item.get("earnings_date") if item else None,
        "eps_surprise_pct":     None,
        "revenue_surprise_pct": None,
        "error":                None,
        "source":               "yfinance",
    }

    try:
        import yfinance as yf
        tk_obj = yf.Ticker(ticker)

        # ── EPS 서프라이즈: earnings_dates DataFrame ──────────────────
        try:
            ed = tk_obj.earnings_dates
            if ed is not None and not ed.empty:
                reported = ed.dropna(subset=["Reported EPS"])
                if not reported.empty:
                    recent_row = reported.iloc[0]
                    surprise_val = recent_row.get("Surprise(%)")
                    if surprise_val is not None:
                        try:
                            base["eps_surprise_pct"] = float(surprise_val)
                        except (TypeError, ValueError):
                            pass
                    if base["latest_earnings_date"] is None:
                        try:
                            base["latest_earnings_date"] = str(reported.index[0].date())
                        except Exception:
                            pass
        except Exception as _exc:
            logger.debug("earnings_dates 조회 실패 (%s): %s", ticker, _exc)

        # ── 매출 서프라이즈: quarterly_income_stmt (추정치 비교 불가; skip for v1) ──
        # revenue_surprise_pct는 v1에서 None으로 둔다

        # ── 점수 계산 ──────────────────────────────────────────────────
        eps_s = base["eps_surprise_pct"]
        rev_s = base["revenue_surprise_pct"]
        sc, flags = _score_earnings(eps_s, rev_s)

        if sc is None:
            base["error"] = f"{ticker}: 실적 서프라이즈 데이터 없음"
        else:
            base["earnings_score"] = sc
            base["earnings_flags"] = flags

    except Exception as exc:
        base["error"] = f"{ticker}: 실적 데이터 오류 — {exc}"

    return base
