"""
screener/events.py
KR DART 공시 기반 이벤트/촉매 레이어 v1

on-demand KR 분석에서만 사용한다. 전체 스크리너 점수·backtest에는 영향 없음.
DART_API_KEY 환경변수 없으면 event_score=None 반환 — 기존 점수 변경하지 않는다.
"""
import io
import json
import logging
import os
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DART_API_KEY_ENV = "DART_API_KEY"
DART_BASE_URL    = "https://opendart.fss.or.kr/api"
_REQUEST_TIMEOUT = 12  # 초

# ── corp_code 캐시 ─────────────────────────────────────────────────────
# stock_code(6자리) → corp_code(8자리) in-memory 맵
_CORP_CODE_MAP: dict[str, str] = {}
# 로컬 파일 캐시 경로 (.dart_cache/ 는 .gitignore에 포함)
_CORP_CODE_CACHE_FILE = Path(__file__).parent.parent / ".dart_cache" / "corp_code_map.json"
_CORP_CODE_CACHE_TTL_DAYS = 7

# ── 이벤트 분류 키워드 ────────────────────────────────────────────────
# Space-stripped before comparison; keywords may include spaces.

_POSITIVE_KW = (
    "잠정실적", "영업실적", "공급계약", "수주",
    "자사주취득", "자사주 취득", "자기주식취득", "자기주식 취득",
    "배당결정", "무상증자", "신규시설투자", "투자결정", "타법인주식취득",
    "계약체결", "수출계약",
)

_NEGATIVE_KW = (
    "유상증자", "전환사채", "신주인수권부사채", "교환사채", "감자",
    "관리종목", "불성실공시", "횡령", "배임", "소송제기", "감사의견",
    "상장폐지", "최대주주변경", "단기과열", "투자경고", "투자위험",
)

# ── Ownership report patterns (v2) ────────────────────────────────────
# Partial match against space-stripped report_nm.
_OWNERSHIP_REPORTS = (
    "대량보유상황보고서",        # 주식등의대량보유상황보고서(일반/약식)
    "소유주식변동신고서",        # 최대주주등소유주식변동신고서
    "특정증권등소유상황보고서",   # 임원ㆍ주요주주특정증권등소유상황보고서
)

# Direction keywords found inside ownership report names.
_OWNERSHIP_POS_KW = ("취득", "매수", "증가")
_OWNERSHIP_NEG_KW = ("처분", "매도", "감소")


# ─── 공개 API 함수 ─────────────────────────────────────────────────────


def classify_dart_event(report_nm: str) -> str:
    """
    Report name → 'positive' | 'negative' | 'neutral' | 'watch' |
                   'ownership_positive' | 'ownership_negative'.

    Priority: negative > ownership (with direction) > positive > neutral.
    Ownership reports with unknown direction return 'watch'.
    """
    nm = (report_nm or "").replace(" ", "")

    # 1. Negative always takes priority.
    for kw in _NEGATIVE_KW:
        if kw.replace(" ", "") in nm:
            return "negative"

    # 2. Ownership reports — check direction from report name.
    for pattern in _OWNERSHIP_REPORTS:
        if pattern in nm:
            for kw in _OWNERSHIP_POS_KW:
                if kw in nm:
                    return "ownership_positive"
            for kw in _OWNERSHIP_NEG_KW:
                if kw in nm:
                    return "ownership_negative"
            return "watch"  # direction unknown

    # 3. General positive keywords.
    for kw in _POSITIVE_KW:
        if kw.replace(" ", "") in nm:
            return "positive"

    return "neutral"


def score_dart_events(classified: list[dict]) -> int:
    """
    Classified event list → 0-100 score.

    positive:           +10/event, cap +25
    negative:           -15/event, cap -35
    ownership_positive: +7/event,  cap +10  (lighter weight)
    ownership_negative: -7/event,  cap -10  (lighter weight)
    watch:              no score change (flagged only)
    base: 50
    """
    pos     = sum(1 for e in classified if e["classification"] == "positive")
    neg     = sum(1 for e in classified if e["classification"] == "negative")
    own_pos = sum(1 for e in classified if e["classification"] == "ownership_positive")
    own_neg = sum(1 for e in classified if e["classification"] == "ownership_negative")

    score = 50
    score += min(25, pos * 10)
    score -= min(35, neg * 15)
    score += min(10, own_pos * 7)
    score -= min(10, own_neg * 7)
    return max(0, min(100, score))


# ─── DART corp_code 해석 ──────────────────────────────────────────────


def _ensure_corp_code_map(api_key: str) -> dict[str, str]:
    """
    DART corpCode.xml ZIP을 이용해 stock_code → corp_code 매핑을 반환한다.
    조회 순서: 메모리 캐시 → 로컬 파일(.dart_cache/) → DART API 다운로드.
    다운로드한 맵은 로컬 파일에 저장해 재사용(TTL 7일).
    """
    global _CORP_CODE_MAP
    if _CORP_CODE_MAP:
        return _CORP_CODE_MAP

    # 로컬 파일 캐시 확인
    try:
        if _CORP_CODE_CACHE_FILE.exists():
            age_days = (time.time() - _CORP_CODE_CACHE_FILE.stat().st_mtime) / 86400
            if age_days < _CORP_CODE_CACHE_TTL_DAYS:
                loaded = json.loads(_CORP_CODE_CACHE_FILE.read_text(encoding="utf-8"))
                if loaded:
                    _CORP_CODE_MAP = loaded
                    logger.debug("DART corp_code 맵: 로컬 캐시 %d건", len(_CORP_CODE_MAP))
                    return _CORP_CODE_MAP
    except Exception as exc:
        logger.debug("DART corp_code 파일 캐시 읽기 실패: %s", exc)

    # DART API에서 corpCode.xml ZIP 다운로드
    resp = requests.get(
        f"{DART_BASE_URL}/corpCode.xml",
        params={"crtfc_key": api_key},
        timeout=30,
    )
    resp.raise_for_status()

    # ZIP 해제 & XML 파싱
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_data = zf.read("CORPCODE.xml")

    root = ET.fromstring(xml_data)
    mapping: dict[str, str] = {}
    for item in root.iter("list"):
        sc = (item.findtext("stock_code") or "").strip()
        cc = (item.findtext("corp_code") or "").strip()
        if sc and len(sc) == 6 and cc:
            mapping[sc] = cc

    _CORP_CODE_MAP = mapping
    logger.debug("DART corp_code 맵: DART API 다운로드 %d건", len(mapping))

    # 로컬 파일에 저장 (실패해도 무시)
    try:
        _CORP_CODE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CORP_CODE_CACHE_FILE.write_text(
            json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.debug("DART corp_code 캐시 저장 실패: %s", exc)

    return _CORP_CODE_MAP


def _get_corp_code(stock_code: str, api_key: str) -> "str | None":
    """6자리 주식코드 → DART 내부 corp_code(8자리). 없으면 None."""
    try:
        mapping = _ensure_corp_code_map(api_key)
        return mapping.get(stock_code.strip().zfill(6))
    except Exception as exc:
        logger.debug("DART corp_code 조회 실패 (%s): %s", stock_code, exc)
        return None


# ─── DART 공시 목록 ───────────────────────────────────────────────────


def _get_disclosures(corp_code: str, api_key: str, days: int) -> list:
    """DART 공시 목록 (최근 days일)."""
    end_de = date.today()
    bgn_de = end_de - timedelta(days=days)
    try:
        r = requests.get(
            f"{DART_BASE_URL}/list.json",
            params={
                "crtfc_key":  api_key,
                "corp_code":  corp_code,
                "bgn_de":     bgn_de.strftime("%Y%m%d"),
                "end_de":     end_de.strftime("%Y%m%d"),
                "page_count": 20,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        data = r.json()
        if data.get("status") == "000":
            return data.get("list", [])
        if data.get("status") == "013":
            return []  # 조회 결과 없음 — 정상
        logger.debug("DART list 조회: status=%s", data.get("status"))
    except Exception as exc:
        logger.debug("DART 공시목록 조회 실패: %s", exc)
    return []


# ─── 메인 퍼블릭 함수 ─────────────────────────────────────────────────


def fetch_kr_dart_events(code: str, days: int = 30) -> dict:
    """
    KR 종목코드(6자리) → DART 공시 이벤트 조회 + 분류 + 점수.

    반환 dict 키:
      code, events (분류된 공시 목록), event_score (0-100 or None),
      event_flags (표시용 문자열 리스트), positive_events, negative_events,
      neutral_events, error (str or None), source ("DART")

    DART_API_KEY 환경변수 없으면 event_score=None, error 메시지 채움.
    어떤 경우에도 기존 validated_score / final_score / catalyst_score 변경 안 함.
    """
    base: dict = {
        "code":            code,
        "events":          [],
        "event_score":     None,
        "event_flags":     [],
        "positive_events": [],
        "negative_events": [],
        "neutral_events":  [],
        "watch_events":    [],
        "error":           None,
        "source":          "DART",
    }

    api_key = _get_dart_key()
    if not api_key:
        base["error"] = (
            f"DART API 키 없음 — 환경변수 {DART_API_KEY_ENV}를 설정하면 "
            "실제 공시 이벤트를 확인할 수 있습니다. "
            "(https://opendart.fss.or.kr 에서 무료 발급)"
        )
        return base

    # corp_code 조회 (corpCode.xml 맵 사용)
    try:
        corp_code = _get_corp_code(code, api_key)
    except Exception as _exc:
        base["error"] = f"{code}: DART corp_code 조회 중 오류 — {_exc}"
        return base
    if not corp_code:
        base["error"] = (
            f"{code}: DART corp_code를 찾을 수 없습니다 "
            "(상장 종목코드 6자리인지 확인; 비상장·ETF·미등록 종목은 지원 안 됨)"
        )
        return base

    # 공시 목록 조회
    raw = _get_disclosures(corp_code, api_key, days)

    # 분류
    classified: list[dict] = [
        {
            "report_nm":      e.get("report_nm", ""),
            "rcept_dt":       e.get("rcept_dt", ""),
            "classification": classify_dart_event(e.get("report_nm", "")),
            "rcept_no":       e.get("rcept_no", ""),
        }
        for e in raw
    ]

    positives = [e for e in classified if e["classification"] == "positive"]
    negatives = [e for e in classified if e["classification"] == "negative"]
    neutrals  = [e for e in classified if e["classification"] == "neutral"]
    own_pos   = [e for e in classified if e["classification"] == "ownership_positive"]
    own_neg   = [e for e in classified if e["classification"] == "ownership_negative"]
    watches   = [e for e in classified if e["classification"] == "watch"]
    all_watch = own_pos + own_neg + watches  # all ownership-related events

    event_score = score_dart_events(classified) if classified else 50

    flags: list[str] = (
        [f"✅ {e['report_nm']} ({e['rcept_dt']})" for e in positives[:3]]
        + [f"⚠ {e['report_nm']} ({e['rcept_dt']})" for e in negatives[:3]]
        + [f"📈 {e['report_nm']} ({e['rcept_dt']})" for e in own_pos[:2]]
        + [f"📉 {e['report_nm']} ({e['rcept_dt']})" for e in own_neg[:2]]
        + [f"👁 {e['report_nm']} ({e['rcept_dt']})" for e in watches[:3]]
    )

    base.update({
        "events":          classified[:10],
        "event_score":     event_score,
        "event_flags":     flags,
        "positive_events": positives,
        "negative_events": negatives,
        "neutral_events":  neutrals,
        "watch_events":    all_watch,
    })
    return base


def _get_dart_key() -> "str | None":
    return os.environ.get(DART_API_KEY_ENV) or os.environ.get("DART_KEY")
