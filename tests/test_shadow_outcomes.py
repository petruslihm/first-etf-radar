"""
screener/shadow_outcomes.py — Shadow A/B Outcome Tracker tests.
Fully offline: price fetchers are deterministic fakes (dependency-injected).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.shadow_outcomes import (
    append_shadow_ab_snapshot,
    append_shadow_outcome_result,
    build_shadow_ab_snapshot,
    evaluate_snapshot_outcomes,
    group_snapshot_rows_by_segment_fields,
    list_shadow_ab_snapshots,
    list_shadow_outcome_results,
    load_shadow_outcome_results,
    load_shadow_outcome_store,
    save_shadow_outcome_store,
    summarize_segment_snapshot_attribution,
)


# ─── fixtures / fakes ────────────────────────────────────────────────


def _validated_df():
    # order = validated ranking (descending score, but order is what matters)
    return pd.DataFrame([
        {"ticker": "AAA", "name": "Alpha", "validated_score": 80, "price": 100, "market": "US"},
        {"ticker": "BBB", "name": "Bravo", "validated_score": 70, "price": 50,  "market": "US"},
        {"ticker": "CCC", "name": "Charlie", "validated_score": 60, "price": 30, "market": "US"},
    ])


def _shadow_df():
    # shadow_rank order = [BBB, AAA, DDD]
    return pd.DataFrame([
        {"ticker": "BBB", "name": "Bravo", "market": "US", "validated_score": 70,
         "shadow_event_adjusted_score": 78, "shadow_event_delta": 8.0,
         "shadow_event_flags": ["✅ beat"], "shadow_candidate_action": "upgrade_watch",
         "shadow_rank": 1, "original_rank": 2},
        {"ticker": "AAA", "name": "Alpha", "market": "US", "validated_score": 80,
         "shadow_event_adjusted_score": 80, "shadow_event_delta": 0.0,
         "shadow_event_flags": [], "shadow_candidate_action": "unchanged",
         "shadow_rank": 2, "original_rank": 1},
        {"ticker": "DDD", "name": "Delta", "market": "US", "validated_score": 55,
         "shadow_event_adjusted_score": 65, "shadow_event_delta": 10.0,
         "shadow_event_flags": ["✅ x"], "shadow_candidate_action": "upgrade_watch",
         "shadow_rank": 3, "original_rank": 9},
    ])


def _kr_validated_df():
    return pd.DataFrame([
        {"ticker": "005930", "name": "삼성전자", "validated_score": 75, "price": 70000, "market": "KOSPI"},
        {"ticker": "035720", "name": "카카오", "validated_score": 65, "price": 50000, "market": "KOSDAQ"},
    ])


def _kr_shadow_df():
    return pd.DataFrame([
        {"ticker": "005930", "name": "삼성전자", "market": "KOSPI", "validated_score": 75,
         "shadow_event_adjusted_score": 80, "shadow_event_delta": 5.0,
         "shadow_event_flags": ["✅"], "shadow_candidate_action": "upgrade_watch",
         "kr_market_segment": "KOSPI", "benchmark_used": "KOSPI", "shadow_rank": 1},
        {"ticker": "035720", "name": "카카오", "market": "KOSDAQ", "validated_score": 65,
         "shadow_event_adjusted_score": 65, "shadow_event_delta": 0.0,
         "shadow_event_flags": [], "shadow_candidate_action": "unchanged",
         "kr_market_segment": "KOSDAQ", "benchmark_used": "KOSDAQ", "shadow_rank": 2},
    ])


def _make_price_fetcher(series_map=None, default=True):
    """
    series_map: ticker -> list[float] | None.
    default growth: close[i] = 100 + i (so horizon h return = h/100 from entry 100).
    """
    series_map = series_map or {}

    def fn(ticker, market, start_date):
        if ticker in series_map:
            data = series_map[ticker]
            if data is None:
                return None
        elif default:
            data = [100 + i for i in range(25)]
        else:
            return None
        idx = pd.date_range("2026-01-01", periods=len(data), freq="D")
        return pd.Series(data, index=idx)
    return fn


def _make_bench_fetcher(per_day=0.5):
    def fn(label, market, start_date):
        data = [100 + per_day * i for i in range(25)]
        idx = pd.date_range("2026-01-01", periods=len(data), freq="D")
        return pd.Series(data, index=idx)
    return fn


ASOF = datetime(2026, 1, 1, 9, 30, 0)


# ════════════════════════════════════════════════════════════════════
# Snapshot build
# ════════════════════════════════════════════════════════════════════


class TestBuildSnapshot:

    def test_builds_snapshot(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        assert snap["market"] == "US"
        assert snap["top_n"] == 3
        assert snap["asof_date"] == "2026-01-01"
        assert snap["snapshot_id"] == "US_20260101T093000_top3"
        assert len(snap["validated_top"]) == 3
        assert len(snap["shadow_top"]) == 3

    def test_preserves_validated_original_order(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        assert [r["ticker"] for r in snap["validated_top"]] == ["AAA", "BBB", "CCC"]
        assert [r["rank"] for r in snap["validated_top"]] == [1, 2, 3]

    def test_uses_shadow_rank_order(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        assert [r["ticker"] for r in snap["shadow_top"]] == ["BBB", "AAA", "DDD"]

    def test_overlap_add_remove(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        assert set(snap["overlap"]) == {"AAA", "BBB"}
        assert snap["added_by_shadow"] == ["DDD"]
        assert snap["removed_by_shadow"] == ["CCC"]
        assert snap["summary"]["overlap_count"] == 2
        assert snap["summary"]["added_count"] == 1
        assert snap["summary"]["removed_count"] == 1

    def test_entry_price_from_validated(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        amap = {r["ticker"]: r["entry_price"] for r in snap["validated_top"]}
        assert amap["AAA"] == 100.0
        assert amap["BBB"] == 50.0

    def test_missing_price_is_none_not_exception(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        # DDD only exists in shadow, has no price → entry_price None
        ddd = [r for r in snap["shadow_top"] if r["ticker"] == "DDD"][0]
        assert ddd["entry_price"] is None

    def test_summary_avg_delta_and_actions(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        s = snap["summary"]
        # deltas 8, 0, 10 → mean 6
        assert s["avg_shadow_delta"] == 6.0
        assert s["upgrade_watch_count"] == 2
        assert s["downgrade_watch_count"] == 0

    def test_kr_segment_breakdown(self):
        snap = build_shadow_ab_snapshot(_kr_validated_df(), _kr_shadow_df(), "KR",
                                        asof=ASOF, top_n=2)
        assert snap["summary"]["by_segment"] == {"KOSPI": 1, "KOSDAQ": 1}
        # segment tag carried on shadow rows
        segs = {r["ticker"]: r.get("kr_market_segment") for r in snap["shadow_top"]}
        assert segs["005930"] == "KOSPI"
        assert segs["035720"] == "KOSDAQ"

    def test_does_not_mutate_inputs(self):
        vdf, sdf = _validated_df(), _shadow_df()
        vbefore, sbefore = vdf.copy(deep=True), sdf.copy(deep=True)
        build_shadow_ab_snapshot(vdf, sdf, "US", asof=ASOF, top_n=3)
        pd.testing.assert_frame_equal(vdf, vbefore)
        pd.testing.assert_frame_equal(sdf, sbefore)

    def test_json_serializable(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        # must not raise
        s = json.dumps(snap)
        assert isinstance(s, str)

    def test_top_n_limits_rows(self):
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=2)
        assert len(snap["validated_top"]) == 2
        assert len(snap["shadow_top"]) == 2

    def test_explicit_price_col(self):
        df = _validated_df().rename(columns={"price": "현재가"})
        snap = build_shadow_ab_snapshot(df, _shadow_df(), "US",
                                        asof=ASOF, top_n=3, price_col="현재가")
        amap = {r["ticker"]: r["entry_price"] for r in snap["validated_top"]}
        assert amap["AAA"] == 100.0


# ════════════════════════════════════════════════════════════════════
# Snapshot store
# ════════════════════════════════════════════════════════════════════


class TestSnapshotStore:

    def test_load_missing_file_empty(self, tmp_path):
        store = load_shadow_outcome_store(tmp_path / "nope.json")
        assert store == {"version": 1, "snapshots": []}

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "s.json"
        store = {"version": 1, "snapshots": [{"snapshot_id": "X", "market": "US"}]}
        save_shadow_outcome_store(store, path)
        assert load_shadow_outcome_store(path) == store

    def test_append_snapshot(self, tmp_path):
        path = tmp_path / "s.json"
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US",
                                        asof=ASOF, top_n=3)
        res = append_shadow_ab_snapshot(snap, path)
        assert res["status"] == "added"
        assert len(load_shadow_outcome_store(path)["snapshots"]) == 1

    def test_list_by_market(self, tmp_path):
        path = tmp_path / "s.json"
        us = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US", asof=ASOF, top_n=3)
        kr = build_shadow_ab_snapshot(_kr_validated_df(), _kr_shadow_df(), "KR", asof=ASOF, top_n=2)
        append_shadow_ab_snapshot(us, path)
        append_shadow_ab_snapshot(kr, path)
        assert len(list_shadow_ab_snapshots(path, market="US")) == 1
        assert len(list_shadow_ab_snapshots(path, market="KR")) == 1
        assert len(list_shadow_ab_snapshots(path)) == 2

    def test_list_limit(self, tmp_path):
        path = tmp_path / "s.json"
        for i in range(5):
            snap = build_shadow_ab_snapshot(
                _validated_df(), _shadow_df(), "US",
                asof=datetime(2026, 1, 1, 9, i, 0), top_n=3)
            append_shadow_ab_snapshot(snap, path)
        assert len(list_shadow_ab_snapshots(path, limit=2)) == 2

    def test_duplicate_snapshot_deterministic(self, tmp_path):
        path = tmp_path / "s.json"
        snap = build_shadow_ab_snapshot(_validated_df(), _shadow_df(), "US", asof=ASOF, top_n=3)
        r1 = append_shadow_ab_snapshot(snap, path)
        r2 = append_shadow_ab_snapshot(snap, path)
        assert r1["status"] == "added"
        assert r2["status"] == "duplicate"
        assert len(load_shadow_outcome_store(path)["snapshots"]) == 1

    def test_corrupt_store_ignored(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{broken json", encoding="utf-8")
        store = load_shadow_outcome_store(path)
        assert store == {"version": 1, "snapshots": []}


# ════════════════════════════════════════════════════════════════════
# Outcome evaluation
# ════════════════════════════════════════════════════════════════════


def _mini_snapshot():
    rows_v = [
        {"rank": 1, "ticker": "AAA", "name": "A", "market": "US", "entry_price": 100,
         "benchmark_used": "SPY"},
        {"rank": 2, "ticker": "BBB", "name": "B", "market": "US", "entry_price": 100,
         "benchmark_used": "SPY"},
        {"rank": 3, "ticker": "CCC", "name": "C", "market": "US", "entry_price": 100,
         "benchmark_used": "SPY"},
    ]
    rows_s = [
        {"rank": 1, "ticker": "BBB", "name": "B", "market": "US", "entry_price": 100,
         "benchmark_used": "SPY"},
        {"rank": 2, "ticker": "CCC", "name": "C", "market": "US", "entry_price": 100,
         "benchmark_used": "SPY"},
        {"rank": 3, "ticker": "DDD", "name": "D", "market": "US", "entry_price": 100,
         "benchmark_used": "SPY"},
    ]
    return {
        "snapshot_id": "US_20260101T000000_top3", "market": "US",
        "asof_date": "2026-01-01", "top_n": 3,
        "validated_top": rows_v, "shadow_top": rows_s,
        "overlap": ["BBB", "CCC"], "added_by_shadow": ["DDD"],
        "removed_by_shadow": ["AAA"],
        "summary": {},
    }


class TestEvaluateOutcomes:

    def test_basic_5_10_20_returns(self):
        out = evaluate_snapshot_outcomes(_mini_snapshot(), _make_price_fetcher())
        v = out["results"]["validated_top"]
        # entry 100, close[h]=100+h → return h/100
        assert v["5"]["avg_return"] == pytest.approx(0.05)
        assert v["10"]["avg_return"] == pytest.approx(0.10)
        assert v["20"]["avg_return"] == pytest.approx(0.20)
        assert v["5"]["count_available"] == 3
        assert v["5"]["count_missing"] == 0

    def test_missing_exit_price_handled(self):
        # CCC has too-short series → missing exits at all horizons
        fetcher = _make_price_fetcher({"CCC": [100, 101]})
        out = evaluate_snapshot_outcomes(_mini_snapshot(), fetcher)
        v = out["results"]["validated_top"]
        # AAA, BBB available; CCC missing at h=5
        assert v["5"]["count_available"] == 2
        assert v["5"]["count_missing"] == 1

    def test_none_series_handled(self):
        fetcher = _make_price_fetcher({"AAA": None})
        out = evaluate_snapshot_outcomes(_mini_snapshot(), fetcher)
        v = out["results"]["validated_top"]
        assert v["5"]["count_available"] == 2
        assert v["5"]["count_missing"] == 1

    def test_cost_slippage_applied(self):
        out = evaluate_snapshot_outcomes(
            _mini_snapshot(), _make_price_fetcher(),
            cost_bps=100, slippage_bps=50)  # 0.015 total
        v = out["results"]["validated_top"]
        assert v["5"]["avg_return"] == pytest.approx(0.05 - 0.015)

    def test_win_hit_worst_median(self):
        out = evaluate_snapshot_outcomes(_mini_snapshot(), _make_price_fetcher())
        v5 = out["results"]["validated_top"]["5"]
        assert v5["win_rate"] == pytest.approx(1.0)        # all positive
        assert v5["hit_rate_5pct"] == pytest.approx(1.0)   # all >= 0.05
        assert v5["hit_rate_10pct"] == pytest.approx(0.0)  # none >= 0.10 at h=5
        assert v5["worst_return"] == pytest.approx(0.05)
        assert v5["median_return"] == pytest.approx(0.05)

    def test_validated_vs_shadow_groups(self):
        out = evaluate_snapshot_outcomes(_mini_snapshot(), _make_price_fetcher())
        res = out["results"]
        for grp in ("validated_top", "shadow_top", "added_by_shadow",
                    "removed_by_shadow", "overlap", "shadow_minus_validated"):
            assert grp in res

    def test_group_membership_counts(self):
        out = evaluate_snapshot_outcomes(_mini_snapshot(), _make_price_fetcher())
        res = out["results"]
        assert res["added_by_shadow"]["5"]["count_available"] == 1   # DDD
        assert res["removed_by_shadow"]["5"]["count_available"] == 1  # AAA
        assert res["overlap"]["5"]["count_available"] == 2           # BBB, CCC

    def test_shadow_minus_validated_diff(self):
        out = evaluate_snapshot_outcomes(_mini_snapshot(), _make_price_fetcher())
        smv = out["results"]["shadow_minus_validated"]["5"]
        # both groups all return 0.05 → diff 0
        assert smv["avg_return_diff"] == pytest.approx(0.0)

    def test_benchmark_excess_when_fetcher_present(self):
        out = evaluate_snapshot_outcomes(
            _mini_snapshot(), _make_price_fetcher(),
            benchmark_fetcher=_make_bench_fetcher(per_day=0.5))
        v5 = out["results"]["validated_top"]["5"]
        # bench close[h]=100+0.5h → at h=5 bench ret=2.5/100=0.025
        assert v5["benchmark_return"] == pytest.approx(0.025)
        assert v5["excess_return"] == pytest.approx(0.05 - 0.025)
        assert out["params"]["has_benchmark"] is True

    def test_no_benchmark_path(self):
        out = evaluate_snapshot_outcomes(_mini_snapshot(), _make_price_fetcher())
        v5 = out["results"]["validated_top"]["5"]
        assert "benchmark_return" not in v5
        assert out["params"]["has_benchmark"] is False

    def test_result_json_serializable(self):
        out = evaluate_snapshot_outcomes(
            _mini_snapshot(), _make_price_fetcher(),
            benchmark_fetcher=_make_bench_fetcher())
        assert isinstance(json.dumps(out), str)

    def test_horizons_and_params_recorded(self):
        out = evaluate_snapshot_outcomes(
            _mini_snapshot(), _make_price_fetcher(),
            horizons=(5, 10), cost_bps=10, slippage_bps=5)
        assert out["horizons"] == [5, 10]
        assert out["params"]["cost_bps"] == 10
        assert out["params"]["slippage_bps"] == 5


# ════════════════════════════════════════════════════════════════════
# Outcome result store
# ════════════════════════════════════════════════════════════════════


class TestResultStore:

    def _result(self, sid="US_20260101T000000_top3", cost=0.0):
        return evaluate_snapshot_outcomes(
            {**_mini_snapshot(), "snapshot_id": sid},
            _make_price_fetcher(), cost_bps=cost)

    def test_append_and_list(self, tmp_path):
        path = tmp_path / "r.json"
        r = self._result()
        res = append_shadow_outcome_result(r, path)
        assert res["status"] == "added"
        assert len(list_shadow_outcome_results(path)) == 1

    def test_filter_by_snapshot_id(self, tmp_path):
        path = tmp_path / "r.json"
        append_shadow_outcome_result(self._result(sid="S1"), path)
        append_shadow_outcome_result(self._result(sid="S2"), path)
        got = list_shadow_outcome_results(path, snapshot_id="S1")
        assert len(got) == 1
        assert got[0]["snapshot_id"] == "S1"

    def test_corrupt_result_store_safe(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text("not json!!", encoding="utf-8")
        assert load_shadow_outcome_results(path) == {"version": 1, "results": []}
        # append still works after corruption
        res = append_shadow_outcome_result(self._result(), path)
        assert res["status"] == "added"

    def test_duplicate_params_replaced(self, tmp_path):
        path = tmp_path / "r.json"
        append_shadow_outcome_result(self._result(cost=0.0), path)
        r2 = append_shadow_outcome_result(self._result(cost=0.0), path)
        assert r2["status"] == "replaced"
        assert len(list_shadow_outcome_results(path)) == 1

    def test_different_params_added(self, tmp_path):
        path = tmp_path / "r.json"
        append_shadow_outcome_result(self._result(cost=0.0), path)
        append_shadow_outcome_result(self._result(cost=25.0), path)
        assert len(list_shadow_outcome_results(path)) == 2


# ════════════════════════════════════════════════════════════════════
# scoring-contract guard
# ════════════════════════════════════════════════════════════════════


class TestScoringContractGuard:

    def test_validated_score_not_mutated_in_snapshot(self):
        vdf = _validated_df()
        before = vdf["validated_score"].tolist()
        build_shadow_ab_snapshot(vdf, _shadow_df(), "US", asof=ASOF, top_n=3)
        assert vdf["validated_score"].tolist() == before


# ════════════════════════════════════════════════════════════════════
# KR segment outcome attribution
# ════════════════════════════════════════════════════════════════════


def _kr_seg_rows():
    """shadow_top rows carrying KR segment composite fields."""
    return [
        {"rank": 1, "ticker": "005930", "name": "삼성", "market": "KR", "entry_price": 100,
         "benchmark_used": "KOSPI", "kr_market_segment": "KOSPI",
         "kr_segment_shadow_profile": "kospi_trend_supply_v1",
         "kr_segment_candidate_action": "upgrade_watch",
         "kr_segment_shadow_delta": 8.0, "shadow_event_delta": 3.0},
        {"rank": 2, "ticker": "000660", "name": "하이닉스", "market": "KR", "entry_price": 100,
         "benchmark_used": "KOSPI", "kr_market_segment": "KOSPI",
         "kr_segment_shadow_profile": "kospi_trend_supply_v1",
         "kr_segment_candidate_action": "unchanged",
         "kr_segment_shadow_delta": 0.0, "shadow_event_delta": 0.0},
        {"rank": 3, "ticker": "035720", "name": "카카오", "market": "KR", "entry_price": 100,
         "benchmark_used": "KOSDAQ", "kr_market_segment": "KOSDAQ",
         "kr_segment_shadow_profile": "kosdaq_breakout_event_risk_v1",
         "kr_segment_candidate_action": "downgrade_watch",
         "kr_segment_shadow_delta": -10.0, "shadow_event_delta": -8.0},
        {"rank": 4, "ticker": "247540", "name": "에코프로", "market": "KR", "entry_price": 100,
         "benchmark_used": "KOSDAQ", "kr_market_segment": "KOSDAQ",
         "kr_segment_shadow_profile": "kosdaq_breakout_event_risk_v1",
         "kr_segment_candidate_action": "upgrade_watch",
         "kr_segment_shadow_delta": 12.0, "shadow_event_delta": 5.0},
    ]


def _kr_seg_snapshot():
    rows = _kr_seg_rows()
    return {
        "snapshot_id": "KR_20260101T000000_top4", "market": "KR",
        "asof_date": "2026-01-01", "top_n": 4,
        "validated_top": [dict(r) for r in rows], "shadow_top": [dict(r) for r in rows],
        "overlap": [r["ticker"] for r in rows],
        "added_by_shadow": [], "removed_by_shadow": [], "summary": {},
    }


def _kr_seg_fetcher():
    # KOSPI names → +1/day (h5=+5%), KOSDAQ names → +2/day (h5=+10%)
    return _make_price_fetcher({
        "035720": [100 + 2 * i for i in range(25)],
        "247540": [100 + 2 * i for i in range(25)],
    })


class TestSegmentSnapshotBuildIntegration:

    def _dfs(self):
        validated_df = pd.DataFrame([
            {"ticker": "005930", "name": "삼성", "validated_score": 60, "price": 70000, "market": "KOSPI"},
            {"ticker": "035720", "name": "카카오", "validated_score": 58, "price": 50000, "market": "KOSDAQ"},
        ])
        shadow_df = pd.DataFrame([
            {"ticker": "005930", "shadow_event_adjusted_score": 65, "shadow_event_delta": 5,
             "shadow_event_flags": ["✅"], "shadow_rank": 1, "kr_market_segment": "KOSPI"},
            {"ticker": "035720", "shadow_event_adjusted_score": 58, "shadow_event_delta": 0,
             "shadow_event_flags": [], "shadow_rank": 2, "kr_market_segment": "KOSDAQ"},
        ])
        segment_df = pd.DataFrame([
            {"ticker": "005930", "kr_segment_shadow_score": 73, "kr_segment_shadow_delta": 13,
             "kr_segment_shadow_profile": "kospi_trend_supply_v1",
             "kr_segment_candidate_action": "upgrade_watch"},
            {"ticker": "035720", "kr_segment_shadow_score": 48, "kr_segment_shadow_delta": -10,
             "kr_segment_shadow_profile": "kosdaq_breakout_event_risk_v1",
             "kr_segment_candidate_action": "downgrade_watch"},
        ])
        return validated_df, shadow_df, segment_df

    def test_segment_df_adds_fields_to_shadow_rows(self):
        validated_df, shadow_df, segment_df = self._dfs()
        snap = build_shadow_ab_snapshot(validated_df, shadow_df, "KR",
                                        asof=ASOF, top_n=2, segment_df=segment_df)
        r = [x for x in snap["shadow_top"] if x["ticker"] == "005930"][0]
        assert r["kr_segment_shadow_score"] == 73
        assert r["kr_segment_shadow_delta"] == 13
        assert r["kr_segment_shadow_profile"] == "kospi_trend_supply_v1"
        assert r["kr_segment_candidate_action"] == "upgrade_watch"

    def test_no_segment_df_unchanged(self):
        validated_df, shadow_df, _ = self._dfs()
        snap = build_shadow_ab_snapshot(validated_df, shadow_df, "KR",
                                        asof=ASOF, top_n=2)
        r = snap["shadow_top"][0]
        assert "kr_segment_shadow_score" not in r

    def test_segment_snapshot_json_serializable(self):
        validated_df, shadow_df, segment_df = self._dfs()
        snap = build_shadow_ab_snapshot(validated_df, shadow_df, "KR",
                                        asof=ASOF, top_n=2, segment_df=segment_df)
        assert isinstance(json.dumps(snap), str)


class TestGroupSnapshotRowsBySegment:

    def test_groups_by_segment(self):
        g = group_snapshot_rows_by_segment_fields(_kr_seg_rows())
        assert set(g["by_segment"].keys()) == {"KOSPI", "KOSDAQ"}
        assert len(g["by_segment"]["KOSPI"]) == 2
        assert len(g["by_segment"]["KOSDAQ"]) == 2

    def test_groups_by_profile(self):
        g = group_snapshot_rows_by_segment_fields(_kr_seg_rows())
        assert len(g["by_profile"]["kospi_trend_supply_v1"]) == 2
        assert len(g["by_profile"]["kosdaq_breakout_event_risk_v1"]) == 2

    def test_groups_by_action(self):
        g = group_snapshot_rows_by_segment_fields(_kr_seg_rows())
        assert len(g["by_action"]["upgrade_watch"]) == 2
        assert len(g["by_action"]["downgrade_watch"]) == 1
        assert len(g["by_action"]["unchanged"]) == 1

    def test_groups_by_segment_action(self):
        g = group_snapshot_rows_by_segment_fields(_kr_seg_rows())
        assert len(g["by_segment_action"]["KOSPI"]["upgrade_watch"]) == 1
        assert len(g["by_segment_action"]["KOSDAQ"]["downgrade_watch"]) == 1
        assert len(g["by_segment_action"]["KOSDAQ"]["upgrade_watch"]) == 1

    def test_missing_fields_fallback(self):
        rows = [{"ticker": "X"}, {"ticker": "Y", "kr_market_segment": "KOSPI"}]
        g = group_snapshot_rows_by_segment_fields(rows)
        assert "UNKNOWN" in g["by_segment"]
        assert "unclassified" in g["by_profile"]
        assert "unclassified" in g["by_action"]

    def test_empty_input(self):
        g = group_snapshot_rows_by_segment_fields([])
        assert g == {"by_segment": {}, "by_profile": {}, "by_action": {}, "by_segment_action": {}}

    def test_does_not_mutate_rows(self):
        import copy
        rows = _kr_seg_rows()
        before = copy.deepcopy(rows)
        group_snapshot_rows_by_segment_fields(rows)
        assert rows == before

    def test_json_serializable(self):
        g = group_snapshot_rows_by_segment_fields(_kr_seg_rows())
        assert isinstance(json.dumps(g), str)


class TestSummarizeSegmentAttribution:

    def test_counts_by_segment_profile_action(self):
        s = summarize_segment_snapshot_attribution(_kr_seg_snapshot())
        assert s["has_segment_fields"] is True
        assert s["counts"]["by_segment"] == {"KOSPI": 2, "KOSDAQ": 2}
        assert s["counts"]["by_action"]["upgrade_watch"] == 2
        assert s["counts"]["by_segment_action"]["KOSDAQ|downgrade_watch"] == 1

    def test_avg_deltas_by_group(self):
        s = summarize_segment_snapshot_attribution(_kr_seg_snapshot())
        # KOSPI seg deltas: 8, 0 → 4 ; KOSDAQ: -10, 12 → 1
        assert s["avg_kr_segment_shadow_delta"]["by_segment"]["KOSPI"] == 4.0
        assert s["avg_kr_segment_shadow_delta"]["by_segment"]["KOSDAQ"] == 1.0
        # event deltas KOSPI: 3, 0 → 1.5 ; KOSDAQ: -8, 5 → -1.5
        assert s["avg_shadow_event_delta"]["by_segment"]["KOSPI"] == 1.5
        assert s["avg_shadow_event_delta"]["by_segment"]["KOSDAQ"] == -1.5

    def test_top_upgrades_downgrades(self):
        s = summarize_segment_snapshot_attribution(_kr_seg_snapshot())
        ups = [r["ticker"] for r in s["top_upgrades"]]
        assert ups[0] == "247540"  # delta +12 first
        assert "005930" in ups     # delta +8
        downs = [r["ticker"] for r in s["top_downgrades"]]
        assert downs[0] == "035720"  # delta -10

    def test_handles_missing_data(self):
        # US snapshot with no segment fields → empty, clearly-flagged summary
        s = summarize_segment_snapshot_attribution(_mini_snapshot())
        assert s["has_segment_fields"] is False
        assert s["counts"]["by_segment"] == {}
        assert s["top_upgrades"] == []

    def test_handles_empty_snapshot(self):
        s = summarize_segment_snapshot_attribution({})
        assert s["has_segment_fields"] is False
        assert s["count_rows"] == 0

    def test_json_serializable(self):
        s = summarize_segment_snapshot_attribution(_kr_seg_snapshot())
        assert isinstance(json.dumps(s), str)


class TestEvaluateSegmentAttribution:

    def test_kospi_vs_kosdaq_returns_differ(self):
        out = evaluate_snapshot_outcomes(_kr_seg_snapshot(), _kr_seg_fetcher())
        seg = out["results"]["segment_attribution"]["by_segment"]
        assert seg["KOSPI"]["5"]["avg_return"] == pytest.approx(0.05)
        assert seg["KOSDAQ"]["5"]["avg_return"] == pytest.approx(0.10)

    def test_profile_attribution(self):
        out = evaluate_snapshot_outcomes(_kr_seg_snapshot(), _kr_seg_fetcher())
        prof = out["results"]["segment_attribution"]["by_profile"]
        assert "kospi_trend_supply_v1" in prof
        assert "kosdaq_breakout_event_risk_v1" in prof
        assert prof["kosdaq_breakout_event_risk_v1"]["5"]["avg_return"] == pytest.approx(0.10)

    def test_action_attribution(self):
        out = evaluate_snapshot_outcomes(_kr_seg_snapshot(), _kr_seg_fetcher())
        act = out["results"]["segment_attribution"]["by_action"]
        assert set(act.keys()) == {"upgrade_watch", "downgrade_watch", "unchanged"}
        # upgrade_watch = 005930 (KOSPI 0.05) + 247540 (KOSDAQ 0.10) → mean 0.075
        assert act["upgrade_watch"]["5"]["avg_return"] == pytest.approx(0.075)

    def test_segment_action_attribution(self):
        out = evaluate_snapshot_outcomes(_kr_seg_snapshot(), _kr_seg_fetcher())
        sa = out["results"]["segment_attribution"]["by_segment_action"]
        assert "KOSPI|upgrade_watch" in sa
        assert "KOSDAQ|downgrade_watch" in sa
        assert sa["KOSDAQ|downgrade_watch"]["5"]["avg_return"] == pytest.approx(0.10)

    def test_avg_segment_and_event_delta_in_attribution(self):
        out = evaluate_snapshot_outcomes(_kr_seg_snapshot(), _kr_seg_fetcher())
        kospi = out["results"]["segment_attribution"]["by_segment"]["KOSPI"]["5"]
        assert kospi["avg_kr_segment_shadow_delta"] == 4.0
        assert kospi["avg_shadow_event_delta"] == 1.5

    def test_missing_prices_handled(self):
        fetcher = _make_price_fetcher({
            "035720": [100, 101],          # too short → missing exits
            "247540": [100 + 2 * i for i in range(25)],
        })
        out = evaluate_snapshot_outcomes(_kr_seg_snapshot(), fetcher)
        kosdaq = out["results"]["segment_attribution"]["by_segment"]["KOSDAQ"]["5"]
        assert kosdaq["count_available"] == 1   # only 247540
        assert kosdaq["count_missing"] == 1     # 035720

    def test_non_kr_returns_empty_attribution(self):
        out = evaluate_snapshot_outcomes(_mini_snapshot(), _make_price_fetcher())
        assert out["results"]["segment_attribution"] == {}

    def test_attribution_json_serializable(self):
        out = evaluate_snapshot_outcomes(_kr_seg_snapshot(), _kr_seg_fetcher())
        assert isinstance(json.dumps(out), str)

    def test_attribution_survives_result_store_roundtrip(self, tmp_path):
        path = tmp_path / "r.json"
        out = evaluate_snapshot_outcomes(_kr_seg_snapshot(), _kr_seg_fetcher())
        append_shadow_outcome_result(out, path)
        got = list_shadow_outcome_results(path, snapshot_id="KR_20260101T000000_top4")
        assert len(got) == 1
        assert "segment_attribution" in got[0]["results"]
        assert "KOSPI" in got[0]["results"]["segment_attribution"]["by_segment"]
