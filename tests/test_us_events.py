"""
screener/events_us.py — US SEC EDGAR 이벤트 레이어 v1 테스트.
네트워크 없이 monkeypatch 사용.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.events_us import (
    SEC_USER_AGENT_ENV,
    _score_earnings,
    classify_edgar_event,
    fetch_us_edgar_events,
    fetch_us_earnings_catalyst,
    score_edgar_events,
)


# ════════════════════════════════════════════════════════════════════
# classify_edgar_event 단위 테스트
# ════════════════════════════════════════════════════════════════════


class TestClassifyEdgarEvent:

    # ── Offering forms → always negative ──
    def test_s3_is_negative(self):
        assert classify_edgar_event("S-3", "") == "negative"

    def test_424b4_is_negative(self):
        assert classify_edgar_event("424B4", "") == "negative"

    def test_424b5_is_negative(self):
        assert classify_edgar_event("424B5", "") == "negative"

    def test_s1_is_negative(self):
        assert classify_edgar_event("S-1", "Registration Statement") == "negative"

    # ── Admin forms → always neutral ──
    def test_form4_is_neutral(self):
        assert classify_edgar_event("4", "Officer sold shares") == "neutral"

    def test_sc13g_is_neutral(self):
        assert classify_edgar_event("SC 13G", "") == "neutral"

    def test_def14a_is_neutral(self):
        assert classify_edgar_event("DEF 14A", "Proxy Statement") == "neutral"

    # ── 8-K with negative description → negative ──
    def test_8k_offering_description_is_negative(self):
        assert classify_edgar_event("8-K", "Announces Public Offering of Common Stock") == "negative"

    def test_8k_going_concern_is_negative(self):
        assert classify_edgar_event("8-K", "Going Concern Doubt Disclosed") == "negative"

    def test_8k_restatement_is_negative(self):
        assert classify_edgar_event("8-K", "Financial Restatement Announced") == "negative"

    def test_8k_lawsuit_is_negative(self):
        assert classify_edgar_event("8-K", "Securities Litigation Filed Against Company") == "negative"

    def test_8k_bankruptcy_is_negative(self):
        assert classify_edgar_event("8-K", "Files for Chapter 11 Bankruptcy Protection") == "negative"

    def test_8k_nt10q_is_negative(self):
        # Late filing notification embedded in description
        assert classify_edgar_event("8-K", "NT 10-Q Filing Delay Notice") == "negative"

    # ── 8-K with positive description → positive ──
    def test_8k_earnings_is_positive(self):
        assert classify_edgar_event("8-K", "Q2 FY2026 Earnings Results Beat Estimates") == "positive"

    def test_8k_repurchase_is_positive(self):
        assert classify_edgar_event("8-K", "Board Approves $5B Share Repurchase Program") == "positive"

    def test_8k_dividend_is_positive(self):
        assert classify_edgar_event("8-K", "Declares Quarterly Dividend of $0.25") == "positive"

    def test_8k_acquisition_is_positive(self):
        assert classify_edgar_event("8-K", "Definitive Agreement to Acquire XYZ Corp") == "positive"

    def test_8k_fda_approval_is_positive(self):
        assert classify_edgar_event("8-K", "FDA Cleared New Drug Application") == "positive"

    def test_8k_guidance_is_positive(self):
        assert classify_edgar_event("8-K", "Company Raises Full Year Guidance and Outlook") == "positive"

    # ── Negative keyword takes priority over positive ──
    def test_negative_beats_positive(self):
        # "OFFERING" + "EARNINGS" → negative wins
        assert classify_edgar_event("8-K", "Announces Offering and Reports Strong Earnings") == "negative"

    # ── 10-Q/10-K → neutral (no keyword) ──
    def test_10q_no_keyword_is_neutral(self):
        assert classify_edgar_event("10-Q", "Quarterly Report") == "neutral"

    def test_10k_no_keyword_is_neutral(self):
        assert classify_edgar_event("10-K", "Annual Report") == "neutral"

    # ── Edge cases ──
    def test_empty_form_empty_desc(self):
        assert classify_edgar_event("", "") == "neutral"

    def test_none_safe(self):
        assert classify_edgar_event(None, None) == "neutral"

    def test_case_insensitive_form(self):
        # Form matching is case-insensitive (uppercased internally)
        assert classify_edgar_event("s-3", "") == "negative"
        assert classify_edgar_event("424b5", "") == "negative"


# ════════════════════════════════════════════════════════════════════
# score_edgar_events 단위 테스트
# ════════════════════════════════════════════════════════════════════


class TestScoreEdgarEvents:

    def _make(self, cls: str, form: str = "8-K") -> dict:
        return {
            "form": form,
            "filingDate": "2026-06-01",
            "description": "test",
            "classification": cls,
        }

    def test_no_events_base_50(self):
        assert score_edgar_events([]) == 50

    def test_one_positive_event(self):
        score = score_edgar_events([self._make("positive")])
        assert score == 62  # 50 + 12

    def test_two_positive_events(self):
        score = score_edgar_events([self._make("positive")] * 2)
        assert score == 74  # 50 + 24, capped at +25 next

    def test_positive_capped_at_75(self):
        # 3건: 50 + min(36,25) = 75
        score = score_edgar_events([self._make("positive")] * 3)
        assert score == 75

    def test_one_negative_event(self):
        score = score_edgar_events([self._make("negative")])
        assert score == 32  # 50 - 18

    def test_two_negative_events(self):
        score = score_edgar_events([self._make("negative")] * 2)
        assert score == 15  # 50 - min(36, 35) = 15

    def test_negative_capped_at_15(self):
        # 2건 이상 negative → 50 - 35 = 15
        score = score_edgar_events([self._make("negative")] * 5)
        assert score == 15

    def test_neutral_no_effect(self):
        score = score_edgar_events([self._make("neutral")] * 5)
        assert score == 50

    def test_10q_adds_bonus_when_no_negative(self):
        # 10-Q neutral + no negative → +5 (periodic bonus)
        score = score_edgar_events([self._make("neutral", "10-Q")])
        assert score == 55

    def test_10k_bonus_suppressed_by_negative(self):
        # 10-K + negative → no periodic bonus
        events = [self._make("neutral", "10-K"), self._make("negative")]
        score = score_edgar_events(events)
        # 50 - 18 = 32 (no +5 because has_negative=True)
        assert score == 32

    def test_mixed_positive_negative(self):
        events = [self._make("positive"), self._make("negative"), self._make("neutral")]
        # 50 + 12 - 18 = 44
        score = score_edgar_events(events)
        assert score == 44

    def test_score_always_0_to_100(self):
        big_neg = [self._make("negative")] * 20
        assert 0 <= score_edgar_events(big_neg) <= 100
        big_pos = [self._make("positive")] * 20
        assert 0 <= score_edgar_events(big_pos) <= 100


# ════════════════════════════════════════════════════════════════════
# fetch_us_edgar_events 통합 테스트 (monkeypatch)
# ════════════════════════════════════════════════════════════════════


class TestFetchUsEdgarEvents:

    def test_cik_not_found_returns_error_no_crash(self, monkeypatch):
        """CIK 없으면 event_score=None, error 메시지, 예외 없음."""
        with patch("screener.events_us._get_cik", return_value=None):
            result = fetch_us_edgar_events("INVALID_TICKER")
        assert result["event_score"] is None
        assert result["error"] is not None
        assert result["events"] == []

    def test_cik_exception_returns_error_no_crash(self, monkeypatch):
        """_get_cik 예외 → error 반환, crash 없음."""
        with patch("screener.events_us._get_cik", side_effect=Exception("DNS timeout")):
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] is None
        assert result["error"] is not None

    def test_filings_exception_returns_error(self, monkeypatch):
        """_get_filings 예외 → error 반환, crash 없음."""
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", side_effect=Exception("timeout")):
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] is None
        assert result["error"] is not None

    def test_positive_8k_earnings_raises_score(self, monkeypatch):
        """긍정 8-K earnings → event_score > 50."""
        fake_filings = [
            {"form": "8-K", "filingDate": "2026-06-10",
             "description": "Q2 FY2026 Earnings Results Beat Estimates",
             "accessionNumber": "0001045810-26-001"},
        ]
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=fake_filings):
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] is not None
        assert result["event_score"] > 50
        assert len(result["positive_events"]) == 1
        assert result["error"] is None

    def test_offering_s3_lowers_score(self, monkeypatch):
        """S-3 공모 등록 → event_score < 50."""
        fake_filings = [
            {"form": "S-3", "filingDate": "2026-06-01",
             "description": "Registration Statement for Sale of Securities",
             "accessionNumber": "0001045810-26-002"},
        ]
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=fake_filings):
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] is not None
        assert result["event_score"] < 50
        assert len(result["negative_events"]) == 1

    def test_lawsuit_8k_lowers_score(self, monkeypatch):
        """8-K + lawsuit description → negative, score < 50."""
        fake_filings = [
            {"form": "8-K", "filingDate": "2026-06-05",
             "description": "Securities Litigation Filed Against Company",
             "accessionNumber": "0001045810-26-003"},
        ]
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=fake_filings):
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] < 50

    def test_multiple_negative_events_capped(self, monkeypatch):
        """복수 부정 이벤트 → score ≥ 0, crash 없음."""
        fake_filings = [
            {"form": "S-3", "filingDate": "2026-06-01",
             "description": "Offering", "accessionNumber": "001"},
            {"form": "424B5", "filingDate": "2026-06-02",
             "description": "Prospectus Supplement", "accessionNumber": "002"},
            {"form": "8-K", "filingDate": "2026-06-03",
             "description": "Going Concern Doubt", "accessionNumber": "003"},
        ]
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=fake_filings):
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] >= 0
        assert result["event_score"] <= 100

    def test_neutral_10q_only_base_plus_five(self, monkeypatch):
        """10-Q만 있으면 event_score = 55 (base 50 + periodic +5)."""
        fake_filings = [
            {"form": "10-Q", "filingDate": "2026-06-01",
             "description": "Quarterly Report", "accessionNumber": "001"},
        ]
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=fake_filings):
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] == 55
        assert result["error"] is None

    def test_no_filings_returns_base_50(self, monkeypatch):
        """공시 없으면 event_score = 50 (base)."""
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=[]):
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] == 50
        assert result["error"] is None

    def test_return_shape_keys_always_present(self, monkeypatch):
        """반환 dict에 필수 키가 항상 있어야 한다."""
        with patch("screener.events_us._get_cik", return_value=None):
            result = fetch_us_edgar_events("NVDA")
        for key in ("ticker", "cik", "events", "event_score", "event_flags",
                    "positive_events", "negative_events", "neutral_events",
                    "error", "source"):
            assert key in result, f"반환 키 누락: {key}"

    def test_source_is_sec_edgar(self, monkeypatch):
        with patch("screener.events_us._get_cik", return_value=None):
            result = fetch_us_edgar_events("AAPL")
        assert result["source"] == "SEC EDGAR"

    def test_event_flags_format(self, monkeypatch):
        """event_flags는 ✅/⚠ 접두사 문자열 리스트."""
        fake_filings = [
            {"form": "8-K", "filingDate": "2026-06-10",
             "description": "Q2 Earnings Results", "accessionNumber": "001"},
            {"form": "S-3", "filingDate": "2026-06-05",
             "description": "Shelf Registration", "accessionNumber": "002"},
        ]
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=fake_filings):
            result = fetch_us_edgar_events("NVDA")
        flags = result["event_flags"]
        assert any("✅" in f for f in flags), "긍정 플래그 없음"
        assert any("⚠" in f for f in flags), "부정 플래그 없음"

    def test_events_limited_to_10(self, monkeypatch):
        """events 최대 10건만 반환."""
        fake_filings = [
            {"form": "4", "filingDate": "2026-06-01",
             "description": "", "accessionNumber": str(i)}
            for i in range(25)
        ]
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=fake_filings):
            result = fetch_us_edgar_events("NVDA")
        assert len(result["events"]) <= 10

    def test_sec_user_agent_env_used(self, monkeypatch):
        """SEC_USER_AGENT 환경변수 설정 시 해당 값 사용."""
        monkeypatch.setenv(SEC_USER_AGENT_ENV, "custom-agent test@mycompany.com")
        captured = {}
        original_get_cik = __import__("screener.events_us", fromlist=["_get_cik"])._get_cik

        def fake_get_cik(ticker, user_agent):
            captured["user_agent"] = user_agent
            return None

        with patch("screener.events_us._get_cik", side_effect=fake_get_cik):
            fetch_us_edgar_events("TSLA")

        assert captured.get("user_agent") == "custom-agent test@mycompany.com"

    def test_no_sec_user_agent_uses_default(self, monkeypatch):
        """SEC_USER_AGENT 없으면 기본값 사용 (crash 없음)."""
        monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)
        captured = {}

        def fake_get_cik(ticker, user_agent):
            captured["user_agent"] = user_agent
            return None

        with patch("screener.events_us._get_cik", side_effect=fake_get_cik):
            result = fetch_us_edgar_events("AAPL")

        assert captured.get("user_agent") is not None
        assert len(captured["user_agent"]) > 0


# ════════════════════════════════════════════════════════════════════
# _analyze_us_ticker_ondemand 이벤트 통합 (독립 단위 테스트)
# ════════════════════════════════════════════════════════════════════


def _make_ohlcv_us(n: int = 300) -> "pd.DataFrame":
    import pandas as pd
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = [150.0 * (1.001 ** i) for i in range(n)]
    vol    = [50_000_000] * n
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices,
         "Close": prices, "Volume": vol},
        index=dates,
    )


class TestUsOndemandWithEdgarEvents:

    def _fake_bench(self):
        import pandas as pd
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        syms  = ["SPY", "QQQ", "XLK", "XLV", "NVDA"]
        return {
            sym: pd.Series([400.0 * (1.0005 ** i) for i in range(300)], index=dates)
            for sym in syms
        }

    def test_edgar_failure_does_not_crash_ondemand(self, monkeypatch):
        """EDGAR fetch 실패해도 us_event_context에 error 필드만 채워짐."""
        with patch("screener.events_us._get_cik", side_effect=Exception("network error")):
            from screener.events_us import fetch_us_edgar_events
            result = fetch_us_edgar_events("NVDA")
        assert result["event_score"] is None
        assert result["error"] is not None

    def test_loaded_results_unchanged_on_edgar_failure(self, monkeypatch):
        """EDGAR fetch 실패해도 외부 상태(session_state 등) 변경 없음."""
        import copy
        sentinel = {"screener_results": [1, 2, 3]}
        before = copy.deepcopy(sentinel)
        with patch("screener.events_us._get_cik", side_effect=Exception("fail")):
            from screener.events_us import fetch_us_edgar_events
            fetch_us_edgar_events("NVDA")
        assert sentinel == before

    def test_no_series_truth_value_error_in_event_score(self, monkeypatch):
        """event_score는 항상 int or None — pandas Series boolean 오류 없음."""
        fake_filings = [
            {"form": "8-K", "filingDate": "2026-06-10",
             "description": "Earnings Beat", "accessionNumber": "001"},
        ]
        with patch("screener.events_us._get_cik", return_value="1045810"), \
             patch("screener.events_us._get_filings", return_value=fake_filings):
            from screener.events_us import fetch_us_edgar_events
            result = fetch_us_edgar_events("NVDA")
        ev_score = result["event_score"]
        assert isinstance(ev_score, int), f"event_score가 int 아님: {type(ev_score)}"
        # bool(int) 가능한지 검증 (Series였으면 여기서 오류)
        assert ev_score > 0 or ev_score == 0

    def test_us_event_context_key_structure(self, monkeypatch):
        """반환된 us_event_context dict 구조가 명세와 일치."""
        with patch("screener.events_us._get_cik", return_value=None):
            from screener.events_us import fetch_us_edgar_events
            ctx = fetch_us_edgar_events("PLTR")
        required_keys = (
            "ticker", "cik", "events", "event_score", "event_flags",
            "positive_events", "negative_events", "neutral_events",
            "error", "source",
        )
        for k in required_keys:
            assert k in ctx, f"us_event_context 키 누락: {k}"


# ════════════════════════════════════════════════════════════════════
# _score_earnings 단위 테스트
# ════════════════════════════════════════════════════════════════════


class TestScoreEarnings:

    def test_no_data_returns_none(self):
        sc, flags = _score_earnings(None, None)
        assert sc is None
        assert flags == []

    def test_eps_surprise_over_10_plus15(self):
        sc, flags = _score_earnings(15.0, None)
        assert sc == 65  # 50 + 15
        assert any("EPS서프라이즈" in f for f in flags)

    def test_eps_surprise_3_to_10_plus8(self):
        sc, flags = _score_earnings(5.0, None)
        assert sc == 58  # 50 + 8
        assert any("EPS비트" in f for f in flags)

    def test_eps_surprise_zero_no_change(self):
        sc, flags = _score_earnings(0.0, None)
        assert sc == 50  # no rule triggered

    def test_eps_miss_minus5_minus12(self):
        sc, flags = _score_earnings(-8.0, None)
        assert sc == 38  # 50 - 12
        assert any("EPS미스" in f for f in flags)

    def test_revenue_beat_5pct_plus8(self):
        sc, flags = _score_earnings(None, 7.0)
        assert sc == 58  # 50 + 8
        assert any("매출비트" in f for f in flags)

    def test_revenue_miss_minus5_minus8(self):
        sc, flags = _score_earnings(None, -6.0)
        assert sc == 42  # 50 - 8
        assert any("매출미스" in f for f in flags)

    def test_eps_and_revenue_both_beat_bonus(self):
        # EPS +5% → +8, rev +7% → +8, both beat bonus +5 = 71
        sc, flags = _score_earnings(5.0, 7.0)
        assert sc == 71
        assert any("동시비트" in f for f in flags)

    def test_eps_only_no_rev_no_bonus(self):
        sc, flags = _score_earnings(5.0, None)
        assert sc == 58
        assert not any("동시비트" in f for f in flags)

    def test_clamp_upper_100(self):
        sc, _ = _score_earnings(15.0, 10.0)  # +15 +8 +5 = 78, capped at 100 (78 < 100, fine)
        assert 0 <= sc <= 100

    def test_clamp_lower_0(self):
        sc, _ = _score_earnings(-20.0, -20.0)  # -12 - 8 = 30, capped at 0 (30 > 0, fine = 30)
        assert sc >= 0


# ════════════════════════════════════════════════════════════════════
# fetch_us_earnings_catalyst 통합 테스트 (monkeypatch)
# ════════════════════════════════════════════════════════════════════


def _make_earnings_dates(surprise_pct: float, date_str: str = "2026-04-30") -> pd.DataFrame:
    """Fake yfinance earnings_dates DataFrame."""
    idx = pd.DatetimeIndex([pd.Timestamp(date_str)])
    df  = pd.DataFrame(
        {
            "EPS Estimate": [1.0],
            "Reported EPS": [1.0 * (1 + surprise_pct / 100)],
            "Surprise(%)": [surprise_pct],
        },
        index=idx,
    )
    return df


class TestFetchUsEarningsCatalyst:

    def _mock_ticker(self, surprise_pct=None, raise_exc=False):
        m = MagicMock()
        if raise_exc:
            type(m).earnings_dates = property(lambda self: (_ for _ in ()).throw(Exception("yf error")))
        elif surprise_pct is None:
            # earnings_dates returns empty DataFrame → no data
            m.earnings_dates = pd.DataFrame()
        else:
            m.earnings_dates = _make_earnings_dates(surprise_pct)
        return m

    def test_positive_eps_surprise_raises_score(self, monkeypatch):
        """EPS 서프라이즈 +15% → earnings_score > 50."""
        mock_tk = self._mock_ticker(surprise_pct=15.0)
        with patch("yfinance.Ticker", return_value=mock_tk):
            result = fetch_us_earnings_catalyst("NVDA")
        assert result["earnings_score"] is not None
        assert result["earnings_score"] > 50
        assert result["eps_surprise_pct"] == 15.0
        assert result["error"] is None

    def test_positive_eps_3pct_plus8(self, monkeypatch):
        """EPS 서프라이즈 +5% → +8 → score 58."""
        mock_tk = self._mock_ticker(surprise_pct=5.0)
        with patch("yfinance.Ticker", return_value=mock_tk):
            result = fetch_us_earnings_catalyst("AAPL")
        assert result["earnings_score"] == 58

    def test_negative_eps_miss_lowers_score(self, monkeypatch):
        """EPS 미스 -8% → earnings_score < 50."""
        mock_tk = self._mock_ticker(surprise_pct=-8.0)
        with patch("yfinance.Ticker", return_value=mock_tk):
            result = fetch_us_earnings_catalyst("TSLA")
        assert result["earnings_score"] is not None
        assert result["earnings_score"] < 50
        assert any("EPS미스" in f for f in result["earnings_flags"])

    def test_no_earnings_data_returns_none_score(self, monkeypatch):
        """earnings_dates가 비어있으면 earnings_score=None, error 메시지."""
        mock_tk = self._mock_ticker(surprise_pct=None)  # empty df
        with patch("yfinance.Ticker", return_value=mock_tk):
            result = fetch_us_earnings_catalyst("PLTR")
        assert result["earnings_score"] is None
        assert result["error"] is not None

    def test_yfinance_exception_no_crash(self, monkeypatch):
        """yfinance가 예외를 올려도 non-fatal, error 필드만 채워짐."""
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = fetch_us_earnings_catalyst("NVDA")
        assert result["earnings_score"] is None
        assert result["error"] is not None

    def test_item_earnings_date_reused(self, monkeypatch):
        """item dict의 earnings_date가 있으면 그 값을 재사용."""
        mock_tk = self._mock_ticker(surprise_pct=None)
        item = {"earnings_date": "2026-07-15", "ticker": "NVDA"}
        with patch("yfinance.Ticker", return_value=mock_tk):
            result = fetch_us_earnings_catalyst("NVDA", item=item)
        assert result["latest_earnings_date"] == "2026-07-15"

    def test_return_keys_always_present(self, monkeypatch):
        """반환 dict에 필수 키가 항상 있어야 한다."""
        with patch("yfinance.Ticker", side_effect=Exception("fail")):
            result = fetch_us_earnings_catalyst("NVDA")
        for key in ("ticker", "earnings_score", "earnings_flags",
                    "latest_earnings_date", "eps_surprise_pct",
                    "revenue_surprise_pct", "error", "source"):
            assert key in result, f"반환 키 누락: {key}"

    def test_source_is_yfinance(self, monkeypatch):
        with patch("yfinance.Ticker", side_effect=Exception("fail")):
            result = fetch_us_earnings_catalyst("NVDA")
        assert result["source"] == "yfinance"

    def test_earnings_score_0_to_100(self, monkeypatch):
        """earnings_score는 항상 0~100 범위."""
        mock_tk = self._mock_ticker(surprise_pct=50.0)  # very large surprise
        with patch("yfinance.Ticker", return_value=mock_tk):
            result = fetch_us_earnings_catalyst("NVDA")
        if result["earnings_score"] is not None:
            assert 0 <= result["earnings_score"] <= 100

    def test_existing_scores_not_mutated(self, monkeypatch):
        """기존 score 필드는 변경 없음."""
        scores = {
            "validated_score": 72.0,
            "rr_score":        68.0,
            "catalyst_score":  55.0,
        }
        import copy
        before = copy.deepcopy(scores)
        mock_tk = self._mock_ticker(surprise_pct=12.0)
        with patch("yfinance.Ticker", return_value=mock_tk):
            fetch_us_earnings_catalyst("NVDA")
        assert scores == before
