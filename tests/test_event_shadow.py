"""
screener/event_shadow.py — Top-N batch shadow event enrichment tests.
Fully offline: all fetchers are dependency-injected fakes. No real network.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.event_shadow import (
    build_shadow_event_row,
    calc_event_adjusted_shadow_score,
    combine_us_event_context,
    enrich_shadow_events_for_candidates,
    get_shadow_cache_key,
    infer_kr_market_segment,
    load_shadow_event_cache,
    normalize_kr_event_context,
    save_shadow_event_cache,
)


# ─── fake fetcher factories ──────────────────────────────────────────


def _fake_edgar(score=None, flags=None, error=None):
    def fn(ticker):
        return {
            "ticker": ticker, "event_score": score,
            "event_flags": flags or [], "error": error, "source": "SEC EDGAR",
        }
    return fn


def _fake_earnings(score=None, flags=None, error=None):
    def fn(ticker):
        return {
            "ticker": ticker, "earnings_score": score,
            "earnings_flags": flags or [], "error": error, "source": "yfinance",
        }
    return fn


def _fake_dart(score=None, flags=None, error=None, watch=None):
    def fn(code):
        return {
            "code": code, "event_score": score, "event_flags": flags or [],
            "watch_events": watch or [], "positive_events": [],
            "negative_events": [], "neutral_events": [], "error": error,
            "source": "DART",
        }
    return fn


def _raising_fetcher(exc_msg="boom"):
    def fn(ticker):
        raise RuntimeError(exc_msg)
    return fn


# ════════════════════════════════════════════════════════════════════
# combine_us_event_context
# ════════════════════════════════════════════════════════════════════


class TestCombineUsEventContext:

    def test_both_present_weighted(self):
        edgar = {"event_score": 80, "event_flags": ["✅ a"]}
        earn  = {"earnings_score": 60, "earnings_flags": ["✅ b"]}
        out = combine_us_event_context(edgar, earn)
        # round(0.6*80 + 0.4*60) = round(72) = 72
        assert out["event_score"] == 72
        assert out["edgar_score"] == 80.0
        assert out["earnings_score"] == 60.0

    def test_only_edgar(self):
        out = combine_us_event_context({"event_score": 70}, {"earnings_score": None})
        assert out["event_score"] == 70
        assert out["earnings_score"] is None

    def test_only_earnings(self):
        out = combine_us_event_context({"event_score": None}, {"earnings_score": 65})
        assert out["event_score"] == 65
        assert out["edgar_score"] is None

    def test_neither_present_none(self):
        out = combine_us_event_context({"event_score": None}, {"earnings_score": None})
        assert out["event_score"] is None

    def test_malformed_input_no_crash(self):
        assert combine_us_event_context(None, None)["event_score"] is None
        assert combine_us_event_context("garbage", 123)["event_score"] is None
        assert combine_us_event_context({}, {})["event_score"] is None

    def test_flags_merged(self):
        edgar = {"event_score": 70, "event_flags": ["✅ edgar-pos", "⚠ edgar-neg"]}
        earn  = {"earnings_score": 60, "earnings_flags": ["✅ earn-pos"]}
        out = combine_us_event_context(edgar, earn)
        assert "✅ edgar-pos" in out["event_flags"]
        assert "⚠ edgar-neg" in out["event_flags"]
        assert "✅ earn-pos" in out["event_flags"]
        assert len(out["event_flags"]) == 3

    def test_errors_preserved_separately(self):
        edgar = {"event_score": None, "error": "edgar boom"}
        earn  = {"earnings_score": 60, "error": None}
        out = combine_us_event_context(edgar, earn)
        assert out["error"] == {"edgar": "edgar boom"}

    def test_source_status_reported(self):
        out = combine_us_event_context(
            {"event_score": 70}, {"earnings_score": None, "error": "no data"}
        )
        assert out["source_status"]["edgar"] == "ok"
        assert out["source_status"]["earnings"] == "error"


# ════════════════════════════════════════════════════════════════════
# normalize_kr_event_context
# ════════════════════════════════════════════════════════════════════


class TestNormalizeKrEventContext:

    def test_dart_score_preserved(self):
        out = normalize_kr_event_context({"event_score": 65, "event_flags": ["✅ x"]})
        assert out["event_score"] == 65.0
        assert out["dart_score"] == 65.0

    def test_watch_events_preserved(self):
        watch = [{"report_nm": "대량보유", "classification": "watch"}]
        out = normalize_kr_event_context({"event_score": 50, "watch_events": watch})
        assert out["watch_events"] == watch

    def test_pos_neg_neu_preserved(self):
        ctx = {
            "event_score": 55,
            "positive_events": [{"a": 1}],
            "negative_events": [{"b": 2}],
            "neutral_events": [{"c": 3}],
        }
        out = normalize_kr_event_context(ctx)
        assert out["positive_events"] == [{"a": 1}]
        assert out["negative_events"] == [{"b": 2}]
        assert out["neutral_events"] == [{"c": 3}]

    def test_missing_key_error_is_missing_clean(self):
        ctx = {"event_score": None, "error": "DART API 키 없음 — 환경변수 DART_API_KEY"}
        out = normalize_kr_event_context(ctx)
        assert out["event_score"] is None
        assert out["source_status"]["dart"] == "error"
        assert out["event_summary"] == "DART 키 없음"

    def test_malformed_input_no_crash(self):
        assert normalize_kr_event_context(None)["event_score"] is None
        assert normalize_kr_event_context("garbage")["event_score"] is None
        assert normalize_kr_event_context(42)["event_score"] is None


# ════════════════════════════════════════════════════════════════════
# calc_event_adjusted_shadow_score
# ════════════════════════════════════════════════════════════════════


class TestCalcEventAdjustedShadowScore:

    def test_no_event_score_unchanged(self):
        out = calc_event_adjusted_shadow_score(60.0, {"event_score": None})
        assert out["event_adjusted_score"] == 60.0
        assert out["event_delta"] == 0.0

    def test_none_context_unchanged(self):
        out = calc_event_adjusted_shadow_score(72.0, None)
        assert out["event_adjusted_score"] == 72.0
        assert out["event_delta"] == 0.0

    def test_high_event_score_increases(self):
        out = calc_event_adjusted_shadow_score(60.0, {"event_score": 80})
        assert out["event_adjusted_score"] > 60.0
        assert out["event_delta"] == 6.0

    def test_low_event_score_decreases(self):
        out = calc_event_adjusted_shadow_score(60.0, {"event_score": 20})
        assert out["event_adjusted_score"] < 60.0
        assert out["event_delta"] == -10.0

    def test_positive_flag_contributes(self):
        with_flag = calc_event_adjusted_shadow_score(
            60.0, {"event_score": 80, "event_flags": ["✅ good"]})
        no_flag = calc_event_adjusted_shadow_score(60.0, {"event_score": 80})
        assert with_flag["event_delta"] > no_flag["event_delta"]

    def test_risk_flag_contributes(self):
        out = calc_event_adjusted_shadow_score(
            60.0, {"event_score": 70, "event_flags": ["⚠ bad"]})
        # 65<=70<75 → +3, risk flag -3 → 0
        assert out["event_delta"] == 0.0

    def test_clamp_upper_100(self):
        out = calc_event_adjusted_shadow_score(98.0, {"event_score": 80,
                                                       "event_flags": ["✅ a"]})
        assert out["event_adjusted_score"] <= 100.0

    def test_clamp_lower_0(self):
        out = calc_event_adjusted_shadow_score(2.0, {"event_score": 20})
        assert out["event_adjusted_score"] >= 0.0

    def test_high_risk_extra_penalty(self):
        out = calc_event_adjusted_shadow_score(
            60.0, {"event_score": 40}, row={"top_risk_score": 90})
        # 35<40<65 neutral → 0, then high risk & ev<60 → -3
        assert out["event_delta"] == -3.0
        assert "고위험종목" in out["event_adjustment_reason"]

    def test_row_not_mutated(self):
        row = {"top_risk_score": 90, "validated_score": 60}
        import copy
        before = copy.deepcopy(row)
        calc_event_adjusted_shadow_score(60.0, {"event_score": 80}, row=row)
        assert row == before

    def test_matches_legacy_neutral_returns_base(self):
        # neutral event (35<x<65) keeps base unchanged but non-None
        out = calc_event_adjusted_shadow_score(55.0, {"event_score": 50})
        assert out["event_adjusted_score"] == 55.0
        assert out["event_delta"] == 0.0


# ════════════════════════════════════════════════════════════════════
# infer_kr_market_segment
# ════════════════════════════════════════════════════════════════════


class TestInferKrMarketSegment:

    def test_kospi_from_market_field(self):
        assert infer_kr_market_segment({"market": "KOSPI"}) == "KOSPI"

    def test_kosdaq_from_market_field(self):
        assert infer_kr_market_segment({"market": "KOSDAQ"}) == "KOSDAQ"

    def test_kosdaq_from_ticker_suffix(self):
        assert infer_kr_market_segment({"ticker": "035720.KQ"}) == "KOSDAQ"

    def test_kospi_from_ticker_suffix(self):
        assert infer_kr_market_segment({"ticker": "005930.KS"}) == "KOSPI"

    def test_kospi_from_benchmark_field(self):
        assert infer_kr_market_segment({"benchmark": "코스피"}) == "KOSPI"

    def test_unknown_when_no_signal(self):
        assert infer_kr_market_segment({"ticker": "005930"}) == "UNKNOWN"

    def test_unknown_on_empty_row(self):
        assert infer_kr_market_segment({}) == "UNKNOWN"
        assert infer_kr_market_segment(None) == "UNKNOWN"

    def test_kosdaq_priority_over_kospi_substring(self):
        # bench_name explicitly kosdaq
        assert infer_kr_market_segment({"bench_name": "KOSDAQ150"}) == "KOSDAQ"

    def test_works_on_pandas_series(self):
        s = pd.Series({"market": "KOSDAQ", "ticker": "035720"})
        assert infer_kr_market_segment(s) == "KOSDAQ"


# ════════════════════════════════════════════════════════════════════
# build_shadow_event_row
# ════════════════════════════════════════════════════════════════════


class TestBuildShadowEventRow:

    REQUIRED_KEYS = (
        "ticker", "name", "market", "validated_score", "shadow_event_score",
        "shadow_edgar_score", "shadow_earnings_score", "shadow_dart_score",
        "shadow_event_adjusted_score", "shadow_event_delta", "shadow_event_reason",
        "shadow_event_flags", "shadow_event_status", "shadow_event_error",
        "shadow_event_fetched_at",
    )

    def test_us_success(self):
        row = {"ticker": "nvda", "name": "NVIDIA", "validated_score": 70}
        fetchers = {
            "edgar": _fake_edgar(score=80, flags=["✅ 8-K beat"]),
            "earnings": _fake_earnings(score=60, flags=["✅ EPS"]),
        }
        out = build_shadow_event_row(row, "US", fetchers=fetchers)
        assert out["ticker"] == "NVDA"
        assert out["shadow_edgar_score"] == 80.0
        assert out["shadow_earnings_score"] == 60.0
        assert out["shadow_event_score"] == 72  # weighted
        assert out["shadow_event_status"] == "ok"
        assert out["shadow_event_adjusted_score"] > 70  # uplift from event

    def test_us_all_required_keys(self):
        row = {"ticker": "AAPL", "validated_score": 65}
        out = build_shadow_event_row(
            row, "US",
            fetchers={"edgar": _fake_edgar(70), "earnings": _fake_earnings(None)},
        )
        for k in self.REQUIRED_KEYS:
            assert k in out, f"missing key: {k}"

    def test_us_fetcher_exception_handled(self):
        row = {"ticker": "TSLA", "validated_score": 60}
        fetchers = {
            "edgar": _raising_fetcher("edgar down"),
            "earnings": _raising_fetcher("yf down"),
        }
        out = build_shadow_event_row(row, "US", fetchers=fetchers)
        # both fetchers failed → no event score, status missing/error, no crash
        assert out["shadow_event_score"] is None
        assert out["shadow_event_status"] in ("error", "missing", "unavailable")
        # base preserved
        assert out["shadow_event_adjusted_score"] == 60.0

    def test_kr_success(self):
        row = {"ticker": "005930", "name": "삼성전자",
               "validated_score": 68, "market": "KOSPI"}
        out = build_shadow_event_row(
            row, "KR", fetchers={"dart": _fake_dart(score=75, flags=["✅ 잠정실적"])})
        assert out["ticker"] == "005930"
        assert out["shadow_dart_score"] == 75.0
        assert out["shadow_event_score"] == 75.0
        assert out["kr_market_segment"] == "KOSPI"
        assert out["benchmark_used"] == "KOSPI"

    def test_kr_missing_key_handled(self):
        row = {"ticker": "005930", "validated_score": 68, "market": "KOSDAQ"}
        out = build_shadow_event_row(
            row, "KR",
            fetchers={"dart": _fake_dart(
                score=None, error="DART API 키 없음 — DART_API_KEY 설정")},
        )
        assert out["shadow_event_score"] is None
        assert out["shadow_event_status"] in ("missing", "unavailable")
        assert out["kr_market_segment"] == "KOSDAQ"
        assert out["benchmark_used"] == "KOSDAQ"

    def test_kr_segment_columns_present_on_unknown(self):
        row = {"ticker": "005930", "validated_score": 68}  # no market
        out = build_shadow_event_row(
            row, "KR", fetchers={"dart": _fake_dart(score=50)})
        assert out["kr_market_segment"] == "UNKNOWN"
        assert out["benchmark_used"] == "UNKNOWN"

    def test_kr_ticker_zfill(self):
        row = {"ticker": 5930, "validated_score": 68, "market": "KOSPI"}
        out = build_shadow_event_row(
            row, "KR", fetchers={"dart": _fake_dart(score=50)})
        assert out["ticker"] == "005930"

    def test_does_not_mutate_input_row(self):
        row = {"ticker": "NVDA", "validated_score": 70}
        import copy
        before = copy.deepcopy(row)
        build_shadow_event_row(
            row, "US",
            fetchers={"edgar": _fake_edgar(80), "earnings": _fake_earnings(60)})
        assert row == before


# ════════════════════════════════════════════════════════════════════
# cache helpers
# ════════════════════════════════════════════════════════════════════


class TestCache:

    def test_cache_key_format(self):
        key = get_shadow_cache_key("us", "NVDA", "2026-06-23")
        assert key == "US:NVDA:2026-06-23"

    def test_cache_key_accepts_datetime(self):
        from datetime import datetime
        key = get_shadow_cache_key("KR", "005930", datetime(2026, 6, 23, 10, 30))
        assert key == "KR:005930:2026-06-23"

    def test_load_save_roundtrip(self, tmp_path):
        path = tmp_path / "cache.json"
        cache = {"US:NVDA:2026-06-23": {"ticker": "NVDA", "shadow_event_score": 72}}
        save_shadow_event_cache(cache, path)
        loaded = load_shadow_event_cache(path)
        assert loaded == cache

    def test_corrupt_cache_ignored(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("{not valid json", encoding="utf-8")
        loaded = load_shadow_event_cache(path)
        assert loaded == {}

    def test_missing_cache_file_returns_empty(self, tmp_path):
        loaded = load_shadow_event_cache(tmp_path / "does_not_exist.json")
        assert loaded == {}


# ════════════════════════════════════════════════════════════════════
# enrich_shadow_events_for_candidates
# ════════════════════════════════════════════════════════════════════


def _us_candidate_df(n=6):
    return pd.DataFrame([
        {"ticker": f"TK{i}", "name": f"Name{i}",
         "validated_score": 70 - i, "top_risk_score": 50, "market": ""}
        for i in range(n)
    ])


def _kr_candidate_df(n=4):
    markets = ["KOSPI", "KOSDAQ", "KOSPI", "KOSDAQ"]
    return pd.DataFrame([
        {"ticker": f"{i:06d}", "name": f"종목{i}",
         "validated_score": 70 - i, "top_risk_score": 50,
         "market": markets[i % len(markets)]}
        for i in range(n)
    ])


class TestEnrichBatch:

    def test_does_not_mutate_input_df(self, tmp_path):
        df = _us_candidate_df()
        before = df.copy(deep=True)
        enrich_shadow_events_for_candidates(
            df, "US", limit=10, cache_path=tmp_path / "c.json",
            fetchers={"edgar": _fake_edgar(70), "earnings": _fake_earnings(60)},
        )
        pd.testing.assert_frame_equal(df, before)

    def test_respects_limit(self, tmp_path):
        df = _us_candidate_df(n=10)
        res = enrich_shadow_events_for_candidates(
            df, "US", limit=3, cache_path=tmp_path / "c.json",
            fetchers={"edgar": _fake_edgar(70), "earnings": _fake_earnings(60)},
        )
        assert len(res) == 3

    def test_preserves_original_rank(self, tmp_path):
        df = _us_candidate_df(n=5)
        res = enrich_shadow_events_for_candidates(
            df, "US", limit=5, cache_path=tmp_path / "c.json",
            fetchers={"edgar": _fake_edgar(70), "earnings": _fake_earnings(60)},
        )
        # output in original order
        assert list(res["original_rank"]) == [1, 2, 3, 4, 5]
        assert list(res["ticker"]) == ["TK0", "TK1", "TK2", "TK3", "TK4"]

    def test_computes_shadow_rank(self, tmp_path):
        df = _us_candidate_df(n=5)
        res = enrich_shadow_events_for_candidates(
            df, "US", limit=5, cache_path=tmp_path / "c.json",
            fetchers={"edgar": _fake_edgar(70), "earnings": _fake_earnings(60)},
        )
        assert set(res["shadow_rank"]) == {1, 2, 3, 4, 5}

    def test_computes_rank_delta(self, tmp_path):
        df = _us_candidate_df(n=5)
        res = enrich_shadow_events_for_candidates(
            df, "US", limit=5, cache_path=tmp_path / "c.json",
            fetchers={"edgar": _fake_edgar(70), "earnings": _fake_earnings(60)},
        )
        # delta = original_rank - shadow_rank
        for _, r in res.iterrows():
            assert r["shadow_rank_delta"] == r["original_rank"] - r["shadow_rank"]

    def test_action_upgrade_on_big_positive_event(self, tmp_path):
        # one ticker gets a strong event uplift → upgrade_watch
        df = pd.DataFrame([
            {"ticker": "AAA", "validated_score": 50, "top_risk_score": 50, "market": ""},
            {"ticker": "BBB", "validated_score": 49, "top_risk_score": 50, "market": ""},
        ])

        def edgar_fn(ticker):
            score = 80 if ticker == "BBB" else 50
            flags = ["✅ beat"] if ticker == "BBB" else []
            return {"event_score": score, "event_flags": flags, "error": None}

        res = enrich_shadow_events_for_candidates(
            df, "US", limit=10, cache_path=tmp_path / "c.json",
            fetchers={"edgar": edgar_fn, "earnings": _fake_earnings(None)},
        )
        bbb = res[res["ticker"] == "BBB"].iloc[0]
        assert bbb["shadow_candidate_action"] == "upgrade_watch"

    def test_action_unchanged_when_neutral(self, tmp_path):
        df = _us_candidate_df(n=3)
        res = enrich_shadow_events_for_candidates(
            df, "US", limit=3, cache_path=tmp_path / "c.json",
            fetchers={"edgar": _fake_edgar(50), "earnings": _fake_earnings(None)},
        )
        assert set(res["shadow_candidate_action"]) == {"unchanged"}

    def test_continues_after_individual_failure(self, tmp_path):
        df = _us_candidate_df(n=3)

        def edgar_fn(ticker):
            if ticker == "TK1":
                raise RuntimeError("boom")
            return {"event_score": 70, "event_flags": [], "error": None}

        res = enrich_shadow_events_for_candidates(
            df, "US", limit=3, cache_path=tmp_path / "c.json",
            fetchers={"edgar": edgar_fn, "earnings": _fake_earnings(None)},
        )
        assert len(res) == 3  # all rows present despite TK1 failure
        tk1 = res[res["ticker"] == "TK1"].iloc[0]
        assert tk1["shadow_event_status"] in ("error", "missing", "unavailable")

    def test_force_refresh_bypasses_cache(self, tmp_path):
        path = tmp_path / "c.json"
        # seed cache with a stale row for today's key
        from screener.event_shadow import _date_str
        key = get_shadow_cache_key("US", "TK0", _date_str())
        save_shadow_event_cache(
            {key: {"ticker": "TK0", "shadow_event_score": 999,
                   "shadow_event_adjusted_score": 70, "shadow_event_delta": 0.0,
                   "validated_score": 70, "shadow_event_status": "ok"}}, path)

        df = _us_candidate_df(n=1)
        res = enrich_shadow_events_for_candidates(
            df, "US", limit=1, force_refresh=True, cache_path=path,
            fetchers={"edgar": _fake_edgar(60), "earnings": _fake_earnings(None)},
        )
        # rebuilt → score should be 60, not the stale 999
        assert res.iloc[0]["shadow_event_score"] == 60.0

    def test_cached_result_prevents_fetcher_call(self, tmp_path):
        path = tmp_path / "c.json"
        from screener.event_shadow import _date_str
        key = get_shadow_cache_key("US", "TK0", _date_str())
        save_shadow_event_cache(
            {key: {"ticker": "TK0", "shadow_event_score": 42,
                   "shadow_event_adjusted_score": 70, "shadow_event_delta": 0.0,
                   "validated_score": 70, "shadow_event_status": "ok"}}, path)

        df = _us_candidate_df(n=1)
        res = enrich_shadow_events_for_candidates(
            df, "US", limit=1, force_refresh=False, cache_path=path,
            fetchers={"edgar": _raising_fetcher("should not be called"),
                      "earnings": _raising_fetcher("should not be called")},
        )
        # cache hit → fetcher never called → value from cache (42)
        assert res.iloc[0]["shadow_event_score"] == 42

    def test_kr_batch_has_segment_columns(self, tmp_path):
        df = _kr_candidate_df(n=4)
        res = enrich_shadow_events_for_candidates(
            df, "KR", limit=4, cache_path=tmp_path / "c.json",
            fetchers={"dart": _fake_dart(score=55)},
        )
        assert "kr_market_segment" in res.columns
        assert "benchmark_used" in res.columns
        assert set(res["kr_market_segment"]) <= {"KOSPI", "KOSDAQ", "UNKNOWN"}

    def test_empty_df_returns_empty(self, tmp_path):
        res = enrich_shadow_events_for_candidates(
            pd.DataFrame(), "US", cache_path=tmp_path / "c.json",
            fetchers={"edgar": _fake_edgar(70)},
        )
        assert res.empty

    def test_none_df_returns_empty(self, tmp_path):
        res = enrich_shadow_events_for_candidates(
            None, "US", cache_path=tmp_path / "c.json")
        assert res.empty


# ════════════════════════════════════════════════════════════════════
# scoring-contract guard: shadow layer must not touch base scores
# ════════════════════════════════════════════════════════════════════


class TestScoringContractGuard:

    def test_validated_score_passthrough_unchanged(self, tmp_path):
        df = _us_candidate_df(n=3)
        res = enrich_shadow_events_for_candidates(
            df, "US", limit=3, cache_path=tmp_path / "c.json",
            fetchers={"edgar": _fake_edgar(80, flags=["✅ x"]),
                      "earnings": _fake_earnings(80)},
        )
        # validated_score column equals the original input values, untouched
        merged = res.set_index("ticker")["validated_score"].to_dict()
        assert merged["TK0"] == 70
        assert merged["TK1"] == 69
        assert merged["TK2"] == 68
