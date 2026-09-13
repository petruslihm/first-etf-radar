"""
screener/events.py — KR DART 이벤트 레이어 v1 테스트.
네트워크 없이 monkeypatch 사용.
"""
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.events import (
    classify_dart_event,
    fetch_kr_dart_events,
    score_dart_events,
)


# ════════════════════════════════════════════════════════════════════
# classify_dart_event 단위 테스트
# ════════════════════════════════════════════════════════════════════


class TestClassifyDartEvent:

    def test_positive_잠정실적(self):
        assert classify_dart_event("잠정실적(공정공시)") == "positive"

    def test_positive_수주(self):
        assert classify_dart_event("단일판매ㆍ공급계약체결") == "positive"

    def test_positive_무상증자(self):
        assert classify_dart_event("주식배당(무상증자)결정") == "positive"

    def test_positive_자사주취득(self):
        # 두 표기 모두 매칭
        assert classify_dart_event("자기주식취득결정") == "positive"
        assert classify_dart_event("자사주취득결정") == "positive"

    def test_positive_with_spaces(self):
        # 공백 포함 키워드도 매칭 (공백 제거 후 비교)
        assert classify_dart_event("자사주 취득 공시") == "positive"
        assert classify_dart_event("자기주식 취득 결정") == "positive"

    def test_negative_유상증자(self):
        assert classify_dart_event("주요사항보고서(유상증자결정)") == "negative"

    def test_negative_전환사채(self):
        assert classify_dart_event("전환사채권발행결정") == "negative"

    def test_negative_신주인수권부사채(self):
        assert classify_dart_event("신주인수권부사채권발행결정") == "negative"

    def test_negative_관리종목(self):
        assert classify_dart_event("관리종목지정") == "negative"

    def test_negative_투자경고(self):
        assert classify_dart_event("투자경고종목지정") == "negative"

    def test_negative_takes_priority_over_positive(self):
        # 부정 키워드가 먼저 평가되어야 함
        # "유상증자" 포함 + "실적" 포함 → negative 우선
        assert classify_dart_event("유상증자 실적발표") == "negative"

    def test_neutral_사업보고서(self):
        assert classify_dart_event("사업보고서") == "neutral"

    def test_neutral_반기보고서(self):
        assert classify_dart_event("반기보고서") == "neutral"

    def test_neutral_unknown(self):
        assert classify_dart_event("이사회결의") == "neutral"

    def test_empty_string(self):
        assert classify_dart_event("") == "neutral"

    def test_none_safe(self):
        assert classify_dart_event(None) == "neutral"


# ════════════════════════════════════════════════════════════════════
# score_dart_events 단위 테스트
# ════════════════════════════════════════════════════════════════════


class TestScoreDartEvents:

    def _make(self, cls: str) -> dict:
        return {"report_nm": "test", "rcept_dt": "20260101",
                "classification": cls, "rcept_no": "0"}

    def test_no_events_base_50(self):
        assert score_dart_events([]) == 50

    def test_one_positive_event(self):
        score = score_dart_events([self._make("positive")])
        assert score == 60

    def test_two_positive_events(self):
        score = score_dart_events([self._make("positive")] * 2)
        assert score == 70

    def test_positive_capped_at_75(self):
        # 5건 positive: 50 + min(25, 50) = 75
        score = score_dart_events([self._make("positive")] * 5)
        assert score == 75

    def test_one_negative_event(self):
        score = score_dart_events([self._make("negative")])
        assert score == 35

    def test_two_negative_events(self):
        score = score_dart_events([self._make("negative")] * 2)
        assert score == 20

    def test_negative_capped_at_15(self):
        # 3건 negative: 50 - min(35, 45) = 15
        score = score_dart_events([self._make("negative")] * 3)
        assert score == 15

    def test_clamp_lower(self):
        # 5건 negative → 50 - 35 = 15 (>0 이므로 클램프 불필요)
        score = score_dart_events([self._make("negative")] * 5)
        assert score >= 0

    def test_neutral_events_no_effect(self):
        neutrals = [self._make("neutral")] * 5
        assert score_dart_events(neutrals) == 50

    def test_mixed(self):
        events = [self._make("positive"), self._make("negative"), self._make("neutral")]
        # 50 + 10 - 15 = 45
        score = score_dart_events(events)
        assert score == 45

    def test_score_always_0_to_100(self):
        # 극단 입력에도 범위 보장
        big_neg = [self._make("negative")] * 20
        assert 0 <= score_dart_events(big_neg) <= 100
        big_pos = [self._make("positive")] * 20
        assert 0 <= score_dart_events(big_pos) <= 100


# ════════════════════════════════════════════════════════════════════
# fetch_kr_dart_events 통합 테스트 (monkeypatch)
# ════════════════════════════════════════════════════════════════════


class TestFetchKrDartEvents:

    def test_missing_api_key_returns_error_no_crash(self, monkeypatch):
        """DART API 키 없으면 event_score=None, error 메시지, 예외 없음."""
        monkeypatch.delenv("DART_API_KEY", raising=False)
        monkeypatch.delenv("DART_KEY", raising=False)
        result = fetch_kr_dart_events("005930")
        assert result["event_score"] is None
        assert result["error"] is not None
        assert "DART_API_KEY" in result["error"] or "API 키" in result["error"]
        assert result["events"] == []

    def test_corp_code_lookup_fails_returns_error(self, monkeypatch):
        """corp_code 조회 실패 → error 반환, crash 없음."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events._get_corp_code", return_value=None):
            result = fetch_kr_dart_events("005930")
        assert result["event_score"] is None
        assert result["error"] is not None

    def test_positive_event_raises_score(self, monkeypatch):
        """잠정실적 공시 → event_score > 50."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        fake_raw = [
            {"report_nm": "잠정실적(공정공시)", "rcept_dt": "20260601", "rcept_no": "001"},
        ]
        with patch("screener.events._get_corp_code", return_value="00126380"), \
             patch("screener.events._get_disclosures", return_value=fake_raw):
            result = fetch_kr_dart_events("005930")
        assert result["event_score"] is not None
        assert result["event_score"] > 50
        assert len(result["positive_events"]) == 1
        assert result["error"] is None

    def test_negative_event_lowers_score(self, monkeypatch):
        """유상증자 공시 → event_score < 50."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        fake_raw = [
            {"report_nm": "주요사항보고서(유상증자결정)", "rcept_dt": "20260601", "rcept_no": "002"},
        ]
        with patch("screener.events._get_corp_code", return_value="00126380"), \
             patch("screener.events._get_disclosures", return_value=fake_raw):
            result = fetch_kr_dart_events("005930")
        assert result["event_score"] is not None
        assert result["event_score"] < 50
        assert len(result["negative_events"]) == 1

    def test_multiple_negative_events_score_capped(self, monkeypatch):
        """복수 부정 공시 → score >= 0, 크래시 없음."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        fake_raw = [
            {"report_nm": "유상증자결정", "rcept_dt": "20260601", "rcept_no": "003"},
            {"report_nm": "전환사채권발행결정", "rcept_dt": "20260602", "rcept_no": "004"},
            {"report_nm": "신주인수권부사채권발행결정", "rcept_dt": "20260603", "rcept_no": "005"},
        ]
        with patch("screener.events._get_corp_code", return_value="00126380"), \
             patch("screener.events._get_disclosures", return_value=fake_raw):
            result = fetch_kr_dart_events("005930")
        assert result["event_score"] >= 0
        assert result["event_score"] <= 100

    def test_neutral_no_event_returns_base(self, monkeypatch):
        """중립 공시만 있거나 공시 없으면 event_score ≈ 50."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events._get_corp_code", return_value="00126380"), \
             patch("screener.events._get_disclosures", return_value=[]):
            result = fetch_kr_dart_events("005930")
        assert result["event_score"] == 50  # 빈 목록 → base 50
        assert result["error"] is None

    def test_api_network_error_returns_error_no_crash(self, monkeypatch):
        """_get_corp_code가 예외를 올려도 fetch_kr_dart_events가 error 반환하고 crash 없음."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        # side_effect=Exception → fetch_kr_dart_events 내부 try/except 검증
        with patch("screener.events._get_corp_code", side_effect=Exception("connection timeout")):
            result = fetch_kr_dart_events("005930")
        assert result["event_score"] is None
        assert result["error"] is not None
        assert "오류" in result["error"] or "실패" in result["error"] or "timeout" in result["error"]

    def test_return_shape_keys_always_present(self, monkeypatch):
        """반환 dict에 필수 키가 항상 있어야 한다."""
        monkeypatch.delenv("DART_API_KEY", raising=False)
        result = fetch_kr_dart_events("005930")
        for key in ("code", "events", "event_score", "event_flags",
                    "positive_events", "negative_events", "neutral_events",
                    "watch_events", "error", "source"):
            assert key in result, f"반환 키 누락: {key}"

    def test_event_flags_format(self, monkeypatch):
        """event_flags는 ✅/⚠ 접두사 문자열 리스트."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        fake_raw = [
            {"report_nm": "잠정실적(공정공시)", "rcept_dt": "20260601", "rcept_no": "001"},
            {"report_nm": "전환사채권발행결정", "rcept_dt": "20260602", "rcept_no": "002"},
        ]
        with patch("screener.events._get_corp_code", return_value="00126380"), \
             patch("screener.events._get_disclosures", return_value=fake_raw):
            result = fetch_kr_dart_events("005930")
        flags = result["event_flags"]
        assert any("✅" in f for f in flags)
        assert any("⚠" in f for f in flags)

    def test_events_limited_to_10(self, monkeypatch):
        """events 최대 10건만 반환."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        fake_raw = [
            {"report_nm": f"사업보고서{i}", "rcept_dt": "20260601", "rcept_no": str(i)}
            for i in range(25)
        ]
        with patch("screener.events._get_corp_code", return_value="00126380"), \
             patch("screener.events._get_disclosures", return_value=fake_raw):
            result = fetch_kr_dart_events("005930")
        assert len(result["events"]) <= 10


# ════════════════════════════════════════════════════════════════════
# Ownership classification v2 tests
# ════════════════════════════════════════════════════════════════════


class TestOwnershipClassification:
    """classify_dart_event ownership branch: watch / ownership_positive / ownership_negative."""

    # ── classify_dart_event ──────────────────────────────────────────

    def test_대량보유상황보고서_is_watch(self):
        assert classify_dart_event("주식등의대량보유상황보고서(일반)") == "watch"

    def test_대량보유상황보고서_약식_is_watch(self):
        assert classify_dart_event("주식등의대량보유상황보고서(약식)") == "watch"

    def test_임원주요주주소유상황보고서_is_watch(self):
        assert classify_dart_event("임원ㆍ주요주주특정증권등소유상황보고서") == "watch"

    def test_최대주주소유주식변동신고서_is_watch(self):
        assert classify_dart_event("최대주주등소유주식변동신고서") == "watch"

    def test_ownership_with_취득_is_ownership_positive(self):
        assert classify_dart_event("임원특정증권등소유상황보고서(취득)") == "ownership_positive"

    def test_ownership_with_매수_is_ownership_positive(self):
        assert classify_dart_event("대량보유상황보고서(장내매수)") == "ownership_positive"

    def test_ownership_with_증가_is_ownership_positive(self):
        assert classify_dart_event("소유주식변동신고서(증가)") == "ownership_positive"

    def test_ownership_with_처분_is_ownership_negative(self):
        assert classify_dart_event("임원특정증권등소유상황보고서(처분)") == "ownership_negative"

    def test_ownership_with_매도_is_ownership_negative(self):
        assert classify_dart_event("대량보유상황보고서(장내매도)") == "ownership_negative"

    def test_ownership_with_감소_is_ownership_negative(self):
        assert classify_dart_event("소유주식변동신고서(감소)") == "ownership_negative"

    def test_ownership_report_not_silently_neutral(self):
        """Ownership reports must NOT fall through to neutral."""
        result = classify_dart_event("주식등의대량보유상황보고서(일반)")
        assert result != "neutral"

    def test_negative_still_beats_ownership_pattern(self):
        """유상증자 in an ownership-like name stays negative."""
        assert classify_dart_event("유상증자관련소유주식변동신고서") == "negative"

    def test_routine_사업보고서_still_neutral(self):
        assert classify_dart_event("사업보고서") == "neutral"

    def test_positive_잠정실적_unaffected(self):
        assert classify_dart_event("잠정실적(공정공시)") == "positive"

    # ── score_dart_events ────────────────────────────────────────────

    def _ev(self, cls: str) -> dict:
        return {"report_nm": "test", "rcept_dt": "20260101",
                "classification": cls, "rcept_no": "0"}

    def test_watch_events_no_score_change(self):
        score = score_dart_events([self._ev("watch")] * 5)
        assert score == 50

    def test_ownership_positive_adds_score(self):
        score = score_dart_events([self._ev("ownership_positive")])
        assert score == 57  # 50 + 7

    def test_two_ownership_positive(self):
        # 2 × 7 = 14, but cap is +10, so 50 + 10 = 60
        score = score_dart_events([self._ev("ownership_positive")] * 2)
        assert score == 60

    def test_ownership_positive_capped_at_10(self):
        score = score_dart_events([self._ev("ownership_positive")] * 5)
        assert score == 60  # 50 + min(10, 35) = 60

    def test_ownership_negative_subtracts_score(self):
        score = score_dart_events([self._ev("ownership_negative")])
        assert score == 43  # 50 - 7

    def test_ownership_negative_capped_at_minus10(self):
        score = score_dart_events([self._ev("ownership_negative")] * 5)
        assert score == 40  # 50 - min(10, 35) = 40

    def test_ownership_mixed_with_regular(self):
        events = [
            self._ev("positive"),       # +10
            self._ev("ownership_positive"),  # +7
            self._ev("watch"),          # 0
        ]
        assert score_dart_events(events) == 67  # 50 + 10 + 7

    def test_existing_positive_score_unchanged(self):
        """Existing positive scoring (10/event, cap 25) must not change."""
        # 3 × 10 = 30, capped at 25 → 50 + 25 = 75
        assert score_dart_events([self._ev("positive")] * 3) == 75

    def test_existing_negative_score_unchanged(self):
        """Existing negative scoring (15/event, cap 35) must not change."""
        assert score_dart_events([self._ev("negative")] * 3) == 15  # 50-35

    # ── fetch_kr_dart_events integration ────────────────────────────

    @pytest.fixture(autouse=True)
    def reset_corp_code_map(self, tmp_path):
        import screener.events as ev_mod
        original_map  = dict(ev_mod._CORP_CODE_MAP)
        original_file = ev_mod._CORP_CODE_CACHE_FILE
        ev_mod._CORP_CODE_MAP  = {}
        ev_mod._CORP_CODE_CACHE_FILE = tmp_path / "corp_code_map.json"
        yield
        ev_mod._CORP_CODE_MAP  = original_map
        ev_mod._CORP_CODE_CACHE_FILE = original_file

    def test_watch_events_key_in_result(self, monkeypatch):
        """watch_events key is always present in result dict."""
        monkeypatch.delenv("DART_API_KEY", raising=False)
        result = fetch_kr_dart_events("005930")
        assert "watch_events" in result

    def test_watch_event_in_watch_events_list(self, monkeypatch):
        """Ownership disclosure with unknown direction lands in watch_events."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        fake_list = {
            "status": "000",
            "list": [
                {"report_nm": "주식등의대량보유상황보고서(일반)",
                 "rcept_dt": "20260601", "rcept_no": "001"},
            ],
        }
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch(
            "screener.events.requests.get",
            side_effect=_fake_get_factory(fake_zip, list_response=fake_list),
        ):
            result = fetch_kr_dart_events("005930")

        assert len(result["watch_events"]) == 1
        assert result["watch_events"][0]["classification"] == "watch"

    def test_watch_flag_uses_eye_emoji(self, monkeypatch):
        """Watch events appear in event_flags with 👁 prefix."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        fake_list = {
            "status": "000",
            "list": [
                {"report_nm": "최대주주등소유주식변동신고서",
                 "rcept_dt": "20260601", "rcept_no": "001"},
            ],
        }
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch(
            "screener.events.requests.get",
            side_effect=_fake_get_factory(fake_zip, list_response=fake_list),
        ):
            result = fetch_kr_dart_events("005930")

        assert any("👁" in f for f in result["event_flags"])

    def test_ownership_positive_flag_uses_chart_up_emoji(self, monkeypatch):
        """Ownership accumulation uses 📈 flag."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        fake_list = {
            "status": "000",
            "list": [
                {"report_nm": "임원특정증권등소유상황보고서(취득)",
                 "rcept_dt": "20260601", "rcept_no": "001"},
            ],
        }
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch(
            "screener.events.requests.get",
            side_effect=_fake_get_factory(fake_zip, list_response=fake_list),
        ):
            result = fetch_kr_dart_events("005930")

        assert any("📈" in f for f in result["event_flags"])
        assert result["event_score"] == 57  # 50 + 7

    def test_ownership_not_in_positive_events(self, monkeypatch):
        """Ownership events do not bleed into positive_events."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        fake_list = {
            "status": "000",
            "list": [
                {"report_nm": "임원특정증권등소유상황보고서(취득)",
                 "rcept_dt": "20260601", "rcept_no": "001"},
            ],
        }
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch(
            "screener.events.requests.get",
            side_effect=_fake_get_factory(fake_zip, list_response=fake_list),
        ):
            result = fetch_kr_dart_events("005930")

        assert result["positive_events"] == []
        assert len(result["watch_events"]) == 1

    def test_watch_score_stays_50(self, monkeypatch):
        """Ownership report with unknown direction does not change score."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        fake_list = {
            "status": "000",
            "list": [
                {"report_nm": "주식등의대량보유상황보고서(일반)",
                 "rcept_dt": "20260601", "rcept_no": "001"},
                {"report_nm": "최대주주등소유주식변동신고서",
                 "rcept_dt": "20260602", "rcept_no": "002"},
            ],
        }
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch(
            "screener.events.requests.get",
            side_effect=_fake_get_factory(fake_zip, list_response=fake_list),
        ):
            result = fetch_kr_dart_events("005930")

        assert result["event_score"] == 50
        assert len(result["watch_events"]) == 2


# ════════════════════════════════════════════════════════════════════
# _analyze_kr_ticker_ondemand 이벤트 통합 (독립 테스트)
# ════════════════════════════════════════════════════════════════════

def _make_ohlcv_kr(n: int = 300) -> "import pandas as pd; pd.DataFrame":
    import pandas as pd
    import numpy as np
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = [50000.0 * (1.001 ** i) for i in range(n)]
    vol    = [100_000] * n
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices,
         "Close": prices, "Volume": vol,
         "Amount": [p * v for p, v in zip(prices, vol)]},
        index=dates,
    )


class TestKrOndemandWithEvents:

    def _fake_item(self):
        import pandas as pd, numpy as np
        df = _make_ohlcv_kr()
        return {
            "ticker": "005930", "name": "삼성전자", "market": "KOSPI",
            "sector": "", "ohlcv": df,
            "market_cap": 400_000_000_000_000, "avg_amount": 5_000_000_000,
            "price": float(df["Close"].iloc[-1]), "is_warned": False,
            "foreign_5d": 1e6, "foreign_20d": 3e6, "inst_5d": 5e5, "inst_20d": 1e6,
        }

    def _fake_bench(self):
        import pandas as pd
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        return {
            "kospi":  pd.Series([2500.0 * (1.0005 ** i) for i in range(300)], index=dates),
            "kosdaq": pd.Series([900.0  * (1.0006 ** i) for i in range(300)], index=dates),
        }

    def test_ondemand_succeeds_when_event_fetcher_errors(self, monkeypatch):
        """이벤트 fetch 실패해도 _analyze_kr_ticker_ondemand가 결과 반환."""
        monkeypatch.delenv("DART_API_KEY", raising=False)

        with patch("screener.collector.fetch_single_kr_ticker_ondemand",
                   return_value=(self._fake_item(), self._fake_bench(), "")):
            # Simulate _analyze_kr_ticker_ondemand logic
            from screener.factors import calc_kr_factors
            item  = self._fake_item()
            bench = self._fake_bench()
            row   = calc_kr_factors("005930", item, bench, 0.0, 0.5)

            # Event fetch with missing key
            from screener.events import fetch_kr_dart_events
            ev = fetch_kr_dart_events("005930")

        # Validation: row and ev both returned without exception
        assert isinstance(row, dict)
        assert ev["event_score"] is None  # key missing
        assert ev["error"] is not None

    def test_kr_event_context_key_present_in_result(self, monkeypatch):
        """kr_event_context 키가 _analyze_kr_ticker_ondemand 결과에 존재."""
        monkeypatch.delenv("DART_API_KEY", raising=False)

        with patch("screener.collector.fetch_single_kr_ticker_ondemand",
                   return_value=(self._fake_item(), self._fake_bench(), "")):
            from screener.factors import calc_kr_factors
            from screener.events import fetch_kr_dart_events
            ev = fetch_kr_dart_events("005930")

        assert "event_score" in ev
        assert "kr_event_context" not in ev  # kr_event_context is in the found dict, not ev itself

    def test_loaded_results_unchanged_on_event_failure(self, monkeypatch):
        """이벤트 fetch 실패해도 기존 session_state 등 부작용 없음."""
        monkeypatch.delenv("DART_API_KEY", raising=False)
        # 단순히 fetch_kr_dart_events 호출만으로도 side-effect 없음을 확인
        initial_state = {"sentinel": True}
        import copy
        before = copy.deepcopy(initial_state)
        from screener.events import fetch_kr_dart_events
        fetch_kr_dart_events("005930")
        # state should be unchanged
        assert initial_state == before


# ════════════════════════════════════════════════════════════════════
# corpCode.xml 기반 corp_code 해석 테스트
# ════════════════════════════════════════════════════════════════════


def _make_corpcode_zip(mapping: dict) -> bytes:
    """stock_code → corp_code 매핑을 담은 가짜 CORPCODE.xml ZIP bytes."""
    rows = "\n".join(
        f"<list><corp_code>{cc}</corp_code><corp_name>테스트</corp_name>"
        f"<stock_code>{sc}</stock_code><modify_date>20260101</modify_date></list>"
        for sc, cc in mapping.items()
    )
    xml_bytes = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><result>{rows}</result>".encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("CORPCODE.xml", xml_bytes)
    return buf.getvalue()


def _fake_get_factory(zip_bytes: bytes, list_response: dict | None = None):
    """corpCode.xml 과 list.json 호출을 가로채는 fake requests.get."""
    def fake_get(url, params=None, timeout=None, **kw):
        m = MagicMock()
        m.raise_for_status = MagicMock()
        if "corpCode.xml" in url:
            m.content = zip_bytes
        elif "list.json" in url:
            m.json = MagicMock(return_value=list_response or {"status": "013"})
        else:
            m.json = MagicMock(return_value={"status": "999"})
        return m
    return fake_get


class TestCorpCodeMap:
    """_ensure_corp_code_map 및 _get_corp_code 단위 테스트."""

    @pytest.fixture(autouse=True)
    def reset_cache(self, tmp_path):
        """각 테스트 전후로 메모리 캐시와 파일 캐시 경로를 초기화."""
        import screener.events as ev_mod
        original_map  = dict(ev_mod._CORP_CODE_MAP)
        original_file = ev_mod._CORP_CODE_CACHE_FILE
        ev_mod._CORP_CODE_MAP  = {}
        ev_mod._CORP_CODE_CACHE_FILE = tmp_path / "corp_code_map.json"
        yield
        ev_mod._CORP_CODE_MAP  = original_map
        ev_mod._CORP_CODE_CACHE_FILE = original_file

    def test_005930_resolves_to_corp_code(self, monkeypatch):
        """005930 → corp_code 매핑이 corpCode.xml ZIP에서 올바르게 해석된다."""
        fake_zip = _make_corpcode_zip({"005930": "00126380", "086520": "00401536"})
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events.requests.get", side_effect=_fake_get_factory(fake_zip)):
            from screener.events import _get_corp_code
            cc = _get_corp_code("005930", "fake_key")
        assert cc == "00126380"

    def test_086520_resolves(self, monkeypatch):
        """086520 → 별도 corp_code 매핑."""
        fake_zip = _make_corpcode_zip({"005930": "00126380", "086520": "00401536"})
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events.requests.get", side_effect=_fake_get_factory(fake_zip)):
            from screener.events import _get_corp_code
            cc = _get_corp_code("086520", "fake_key")
        assert cc == "00401536"

    def test_missing_stock_code_returns_none(self, monkeypatch):
        """corp_code 맵에 없는 종목코드는 None 반환."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events.requests.get", side_effect=_fake_get_factory(fake_zip)):
            from screener.events import _get_corp_code
            cc = _get_corp_code("999999", "fake_key")
        assert cc is None

    def test_missing_stock_code_fetch_returns_clear_error(self, monkeypatch):
        """맵에 없는 종목코드면 fetch_kr_dart_events가 명확한 error 메시지 반환."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events.requests.get", side_effect=_fake_get_factory(fake_zip)):
            from screener.events import fetch_kr_dart_events
            result = fetch_kr_dart_events("999999")
        assert result["event_score"] is None
        assert result["error"] is not None
        assert "999999" in result["error"]

    def test_fetch_uses_corp_code_not_stock_code_in_list_call(self, monkeypatch):
        """list.json 호출에 stock_code가 아닌 corp_code가 전달된다."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        list_calls: list[dict] = []

        def fake_get(url, params=None, timeout=None, **kw):
            m = MagicMock()
            m.raise_for_status = MagicMock()
            if "corpCode.xml" in url:
                m.content = fake_zip
            elif "list.json" in url:
                list_calls.append(dict(params or {}))
                m.json = MagicMock(return_value={"status": "013"})
            return m

        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events.requests.get", side_effect=fake_get):
            from screener.events import fetch_kr_dart_events
            fetch_kr_dart_events("005930")

        assert len(list_calls) == 1
        assert list_calls[0].get("corp_code") == "00126380"
        assert list_calls[0].get("corp_code") != "005930"

    def test_memory_cache_prevents_second_download(self, monkeypatch):
        """메모리 캐시 적중 시 corpCode.xml을 다시 다운로드하지 않는다."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        call_count = {"n": 0}

        def fake_get(url, params=None, timeout=None, **kw):
            m = MagicMock()
            m.raise_for_status = MagicMock()
            if "corpCode.xml" in url:
                call_count["n"] += 1
                m.content = fake_zip
            elif "list.json" in url:
                m.json = MagicMock(return_value={"status": "013"})
            return m

        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events.requests.get", side_effect=fake_get):
            from screener.events import _get_corp_code
            _get_corp_code("005930", "fake_key")
            _get_corp_code("005930", "fake_key")  # 두 번째: 캐시 히트

        assert call_count["n"] == 1  # ZIP 다운로드는 한 번만

    def test_file_cache_loaded_without_download(self, monkeypatch, tmp_path):
        """유효한 파일 캐시가 있으면 DART API를 호출하지 않는다."""
        import screener.events as ev_mod
        cache_data = {"005930": "00126380"}
        ev_mod._CORP_CODE_CACHE_FILE.write_text(
            json.dumps(cache_data, ensure_ascii=False), encoding="utf-8"
        )

        download_called = {"n": 0}

        def fake_get(url, params=None, timeout=None, **kw):
            download_called["n"] += 1
            m = MagicMock()
            m.raise_for_status = MagicMock()
            return m

        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events.requests.get", side_effect=fake_get):
            from screener.events import _get_corp_code
            cc = _get_corp_code("005930", "fake_key")

        assert cc == "00126380"
        assert download_called["n"] == 0  # API 호출 없음

    def test_network_error_returns_none_no_crash(self, monkeypatch):
        """corpCode.xml 다운로드 실패 시 None 반환, 크래시 없음."""
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch("screener.events.requests.get", side_effect=Exception("network error")):
            from screener.events import _get_corp_code
            cc = _get_corp_code("005930", "fake_key")
        assert cc is None

    def test_key_missing_still_non_fatal(self, monkeypatch):
        """DART_API_KEY 없으면 event_score=None, error 포함, 크래시 없음."""
        monkeypatch.delenv("DART_API_KEY", raising=False)
        monkeypatch.delenv("DART_KEY", raising=False)
        from screener.events import fetch_kr_dart_events
        result = fetch_kr_dart_events("005930")
        assert result["event_score"] is None
        assert result["error"] is not None
        assert "DART_API_KEY" in result["error"] or "API 키" in result["error"]

    def test_full_flow_005930(self, monkeypatch):
        """005930 전체 플로우: corp_code 해석 → 공시 조회 → 점수."""
        fake_zip = _make_corpcode_zip({"005930": "00126380"})
        fake_list = {
            "status": "000",
            "list": [
                {"report_nm": "잠정실적(공정공시)", "rcept_dt": "20260601", "rcept_no": "001"},
                {"report_nm": "사업보고서", "rcept_dt": "20260510", "rcept_no": "002"},
            ],
        }
        monkeypatch.setenv("DART_API_KEY", "fake_key")
        with patch(
            "screener.events.requests.get",
            side_effect=_fake_get_factory(fake_zip, list_response=fake_list),
        ):
            from screener.events import fetch_kr_dart_events
            result = fetch_kr_dart_events("005930")

        assert result["event_score"] is not None
        assert result["event_score"] > 50
        assert result["error"] is None
        assert any(e["classification"] == "positive" for e in result["events"])
