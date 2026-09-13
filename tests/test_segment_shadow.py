"""
screener/segment_shadow.py — KR Segment Shadow Composite v1 tests.
Fully offline; no network. Shadow-only — asserts no mutation of base scores.
"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.segment_shadow import (
    GENERIC_PROFILE,
    KOSDAQ_PROFILE,
    KOSPI_PROFILE,
    apply_kr_segment_shadow_composite,
    calc_kr_segment_shadow_score,
)


# ─── helpers ─────────────────────────────────────────────────────────


def _kospi_strong():
    return {
        "ticker": "005930", "name": "삼성전자", "market": "KOSPI",
        "validated_score": 60, "leader_score": 80, "momentum_score": 70,
        "supply_score": 70, "sect_rs": 65, "top_risk_score": 30, "volume_score": 60,
    }


def _kosdaq_strong():
    return {
        "ticker": "035720", "name": "카카오", "market": "KOSDAQ",
        "validated_score": 60, "breakout_score": 70, "volume_score": 75,
        "near_high": 70, "momentum_accel": 65, "top_risk_score": 50, "buyable_score": 60,
    }


def _event_row(adj=None, delta=0.0, flags=None, segment=None):
    r = {"shadow_event_adjusted_score": adj, "shadow_event_delta": delta,
         "shadow_event_flags": flags or []}
    if segment is not None:
        r["kr_market_segment"] = segment
    return r


# ════════════════════════════════════════════════════════════════════
# calc_kr_segment_shadow_score
# ════════════════════════════════════════════════════════════════════


class TestCalcKrSegmentShadowScore:

    def test_kospi_strong_positive_delta(self):
        out = calc_kr_segment_shadow_score(_kospi_strong())
        assert out["kr_segment"] == "KOSPI"
        assert out["kr_segment_shadow_profile"] == KOSPI_PROFILE
        # +4 leader +3 mom +4 supply +2 sector = +13 over base 60
        assert out["kr_segment_shadow_score"] == 73.0
        assert out["kr_segment_shadow_delta"] == 13.0

    def test_kospi_weak_trend_negative_event_penalty(self):
        row = {"ticker": "000660", "market": "KOSPI", "validated_score": 60,
               "leader_score": 50, "top_risk_score": 90, "volume_score": 40}
        ev = _event_row(adj=55, delta=-8)
        out = calc_kr_segment_shadow_score(row, ev)
        # base 55 (event-adjusted): -4 risk -5 event -3 vol = -12 → 43
        assert out["kr_segment_shadow_score"] == 43.0
        assert out["kr_segment_shadow_delta"] == -12.0

    def test_kosdaq_breakout_volume_positive_delta(self):
        ev = _event_row(adj=64, delta=3)
        out = calc_kr_segment_shadow_score(_kosdaq_strong(), ev)
        assert out["kr_segment"] == "KOSDAQ"
        assert out["kr_segment_shadow_profile"] == KOSDAQ_PROFILE
        # base 64: +5 brk +4 vol +3 near +2 accel = +14 → 78
        assert out["kr_segment_shadow_score"] == 78.0
        assert out["kr_segment_shadow_delta"] == 14.0

    def test_kosdaq_financing_flag_penalty(self):
        row = {"ticker": "035720", "market": "KOSDAQ", "validated_score": 60}
        ev = _event_row(adj=50, delta=-3,
                        flags=["⚠ 전환사채권발행결정 (20260601)"])
        out = calc_kr_segment_shadow_score(row, ev)
        # only financing flag triggers: -5 → 45
        assert out["kr_segment_shadow_score"] == 45.0
        assert out["kr_segment_shadow_delta"] == -5.0
        assert any("증자/희석" in c["label"] for c in out["kr_segment_shadow_components"])

    def test_kosdaq_english_cb_bw_flag_penalty(self):
        row = {"ticker": "035720", "market": "KOSDAQ", "validated_score": 60}
        ev = _event_row(adj=50, delta=0, flags=["CB issuance announced"])
        out = calc_kr_segment_shadow_score(row, ev)
        assert out["kr_segment_shadow_delta"] == -5.0

    def test_kosdaq_high_risk_penalty(self):
        row = {"ticker": "035720", "market": "KOSDAQ", "validated_score": 60,
               "top_risk_score": 85}
        out = calc_kr_segment_shadow_score(row)
        # -6 risk → 54
        assert out["kr_segment_shadow_score"] == 54.0
        assert out["kr_segment_shadow_delta"] == -6.0

    def test_unknown_generic_fallback(self):
        row = {"ticker": "005930", "market": "", "validated_score": 60}
        ev = _event_row(adj=58, delta=0)
        out = calc_kr_segment_shadow_score(row, ev)
        assert out["kr_segment"] == "UNKNOWN"
        assert out["kr_segment_shadow_profile"] == GENERIC_PROFILE
        assert out["kr_segment_shadow_score"] == 58.0  # generic event-adjusted base
        assert out["kr_segment_shadow_delta"] == 0.0
        assert "UNKNOWN" in out["kr_segment_shadow_reason"]

    def test_segment_from_event_row_preferred(self):
        # row says nothing, event_row tags KOSDAQ
        row = {"ticker": "035720", "validated_score": 60, "breakout_score": 70}
        ev = _event_row(adj=60, delta=0, segment="KOSDAQ")
        out = calc_kr_segment_shadow_score(row, ev)
        assert out["kr_segment"] == "KOSDAQ"

    def test_missing_factor_columns_no_crash(self):
        row = {"ticker": "005930", "market": "KOSPI"}
        out = calc_kr_segment_shadow_score(row)
        # base 50 (no validated, no event), no triggers
        assert out["kr_segment_shadow_score"] == 50.0
        assert out["kr_segment_shadow_delta"] == 0.0

    def test_no_input_mutation(self):
        row = _kospi_strong()
        ev = _event_row(adj=70, delta=5)
        rb, eb = copy.deepcopy(row), copy.deepcopy(ev)
        calc_kr_segment_shadow_score(row, ev)
        assert row == rb
        assert ev == eb

    def test_clamp_upper_100(self):
        row = {**_kospi_strong(), "validated_score": 98}
        out = calc_kr_segment_shadow_score(row)  # base 98 + 13 → clamp 100
        assert out["kr_segment_shadow_score"] == 100.0

    def test_clamp_lower_0(self):
        row = {"ticker": "035720", "market": "KOSDAQ", "validated_score": 3,
               "top_risk_score": 85, "buyable_score": 40}
        ev = _event_row(adj=3, delta=-8, flags=["⚠ 유상증자결정"])
        out = calc_kr_segment_shadow_score(row, ev)
        # base 3: -6 risk -8 event -5 financing -4 buyable = -23 → clamp 0
        assert out["kr_segment_shadow_score"] == 0.0

    def test_components_present_and_json_serializable(self):
        out = calc_kr_segment_shadow_score(_kospi_strong())
        comps = out["kr_segment_shadow_components"]
        assert isinstance(comps, list) and len(comps) >= 1
        for c in comps:
            assert set(c.keys()) == {"label", "delta", "reason"}
        json.dumps(out)  # must not raise

    def test_event_adjusted_base_used_over_validated(self):
        # validated 60 but event-adjusted 70 → base should be 70
        row = {"ticker": "005930", "market": "KOSPI", "validated_score": 60,
               "leader_score": 80}  # +4 only
        ev = _event_row(adj=70, delta=2)
        out = calc_kr_segment_shadow_score(row, ev)
        assert out["kr_segment_shadow_score"] == 74.0  # 70 + 4


# ════════════════════════════════════════════════════════════════════
# apply_kr_segment_shadow_composite
# ════════════════════════════════════════════════════════════════════


def _validated_df():
    return pd.DataFrame([
        _kospi_strong(),  # 005930 KOSPI strong → +13 → 73
        {"ticker": "035720", "name": "카카오", "market": "KOSDAQ",
         "validated_score": 58, "breakout_score": 40, "volume_score": 40,
         "top_risk_score": 85, "buyable_score": 40},  # KOSDAQ weak → -6 -4 = -10 → 48
        {"ticker": "000660", "name": "하이닉스", "market": "KOSPI",
         "validated_score": 55, "leader_score": 50},  # neutral → 55
    ])


class TestApplyBatch:

    def test_preserves_original_order(self):
        out = apply_kr_segment_shadow_composite(_validated_df())
        assert list(out["original_rank"]) == [1, 2, 3]
        assert list(out["ticker"]) == ["005930", "035720", "000660"]

    def test_segment_shadow_rank_computed(self):
        out = apply_kr_segment_shadow_composite(_validated_df())
        rank = dict(zip(out["ticker"], out["segment_shadow_rank"]))
        # scores: 005930=73, 000660=55, 035720=48 → ranks 1,2,3
        assert rank["005930"] == 1
        assert rank["000660"] == 2
        assert rank["035720"] == 3

    def test_rank_delta_correct(self):
        out = apply_kr_segment_shadow_composite(_validated_df())
        for _, r in out.iterrows():
            assert r["segment_shadow_rank_delta"] == r["original_rank"] - r["segment_shadow_rank"]

    def test_actions(self):
        out = apply_kr_segment_shadow_composite(_validated_df())
        act = dict(zip(out["ticker"], out["kr_segment_candidate_action"]))
        assert act["005930"] == "upgrade_watch"    # delta +13
        assert act["035720"] == "downgrade_watch"  # delta -10
        assert act["000660"] == "unchanged"        # delta 0, small rank move

    def test_joins_shadow_df_by_ticker(self):
        shadow_df = pd.DataFrame([
            {"ticker": "005930", "shadow_event_adjusted_score": 70,
             "shadow_event_delta": 10, "shadow_event_flags": ["✅ 잠정실적"]},
        ])
        out = apply_kr_segment_shadow_composite(_validated_df(), shadow_df=shadow_df)
        row = out[out["ticker"] == "005930"].iloc[0]
        assert row["shadow_event_adjusted_score"] == 70
        # base now 70 (event-adjusted) + 13 = 83
        assert row["kr_segment_shadow_score"] == 83.0

    def test_respects_limit(self):
        out = apply_kr_segment_shadow_composite(_validated_df(), limit=2)
        assert len(out) == 2
        assert list(out["ticker"]) == ["005930", "035720"]

    def test_handles_missing_shadow_df(self):
        out = apply_kr_segment_shadow_composite(_validated_df(), shadow_df=None)
        assert len(out) == 3
        assert out[out["ticker"] == "005930"].iloc[0]["shadow_event_adjusted_score"] is None

    def test_handles_missing_and_duplicate_ticker(self):
        df = pd.DataFrame([
            {"ticker": "005930", "name": "A", "market": "KOSPI", "validated_score": 60},
            {"ticker": "005930", "name": "A-dup", "market": "KOSPI", "validated_score": 60},
            {"ticker": None, "name": "noticker", "market": "KOSDAQ", "validated_score": 50},
        ])
        out = apply_kr_segment_shadow_composite(df)  # must not raise
        assert len(out) == 3

    def test_does_not_mutate_inputs(self):
        vdf = _validated_df()
        shadow_df = pd.DataFrame([
            {"ticker": "005930", "shadow_event_adjusted_score": 70,
             "shadow_event_delta": 10, "shadow_event_flags": ["✅"]},
        ])
        vb, sb = vdf.copy(deep=True), shadow_df.copy(deep=True)
        apply_kr_segment_shadow_composite(vdf, shadow_df=shadow_df)
        pd.testing.assert_frame_equal(vdf, vb)
        pd.testing.assert_frame_equal(shadow_df, sb)

    def test_output_json_serializable(self):
        out = apply_kr_segment_shadow_composite(_validated_df())
        json.dumps(out.to_dict("records"))  # must not raise

    def test_output_columns_present(self):
        out = apply_kr_segment_shadow_composite(_validated_df())
        for col in ("original_rank", "segment_shadow_rank", "segment_shadow_rank_delta",
                    "ticker", "name", "kr_segment", "validated_score",
                    "shadow_event_adjusted_score", "shadow_event_delta",
                    "kr_segment_shadow_score", "kr_segment_shadow_delta",
                    "kr_segment_shadow_profile", "kr_segment_shadow_reason",
                    "kr_segment_shadow_components", "shadow_event_flags",
                    "kr_segment_candidate_action"):
            assert col in out.columns

    def test_empty_df_returns_empty(self):
        out = apply_kr_segment_shadow_composite(pd.DataFrame())
        assert out.empty

    def test_none_df_returns_empty(self):
        out = apply_kr_segment_shadow_composite(None)
        assert out.empty

    def test_validated_score_not_mutated(self):
        vdf = _validated_df()
        before = vdf["validated_score"].tolist()
        apply_kr_segment_shadow_composite(vdf)
        assert vdf["validated_score"].tolist() == before


# ════════════════════════════════════════════════════════════════════
# Snapshot integration
# ════════════════════════════════════════════════════════════════════


class TestSnapshotIntegration:

    def test_snapshot_includes_segment_fields_when_present(self):
        from screener.shadow_outcomes import build_shadow_ab_snapshot
        from datetime import datetime

        validated_df = pd.DataFrame([
            {"ticker": "005930", "name": "삼성", "validated_score": 60, "price": 70000, "market": "KOSPI"},
            {"ticker": "035720", "name": "카카오", "validated_score": 58, "price": 50000, "market": "KOSDAQ"},
        ])
        shadow_df = pd.DataFrame([
            {"ticker": "005930", "shadow_event_adjusted_score": 65, "shadow_event_delta": 5,
             "shadow_event_flags": ["✅"], "shadow_rank": 1, "kr_market_segment": "KOSPI",
             "benchmark_used": "KOSPI"},
            {"ticker": "035720", "shadow_event_adjusted_score": 58, "shadow_event_delta": 0,
             "shadow_event_flags": [], "shadow_rank": 2, "kr_market_segment": "KOSDAQ",
             "benchmark_used": "KOSDAQ"},
        ])
        seg_df = apply_kr_segment_shadow_composite(validated_df, shadow_df=shadow_df)

        snap = build_shadow_ab_snapshot(
            validated_df, shadow_df, "KR",
            asof=datetime(2026, 1, 1, 9, 0, 0), top_n=2, segment_df=seg_df)

        # segment fields attached to rows by ticker
        v0 = [r for r in snap["validated_top"] if r["ticker"] == "005930"][0]
        assert "kr_segment_shadow_score" in v0
        assert "kr_segment_candidate_action" in v0
        import json as _json
        _json.dumps(snap)  # still JSON-serializable

    def test_snapshot_without_segment_df_unchanged(self):
        from screener.shadow_outcomes import build_shadow_ab_snapshot
        from datetime import datetime

        validated_df = pd.DataFrame([
            {"ticker": "005930", "name": "삼성", "validated_score": 60, "price": 70000, "market": "KOSPI"},
        ])
        shadow_df = pd.DataFrame([
            {"ticker": "005930", "shadow_event_adjusted_score": 65, "shadow_event_delta": 5,
             "shadow_event_flags": ["✅"], "shadow_rank": 1},
        ])
        snap = build_shadow_ab_snapshot(
            validated_df, shadow_df, "KR",
            asof=datetime(2026, 1, 1, 9, 0, 0), top_n=1)
        v0 = snap["validated_top"][0]
        assert "kr_segment_shadow_score" not in v0


# ════════════════════════════════════════════════════════════════════
# Adverse Event Risk Classifier v2 — segment integration
# ════════════════════════════════════════════════════════════════════


class TestAdverseIntegration:

    def test_adverse_fields_in_calc_output(self):
        out = calc_kr_segment_shadow_score(_kospi_strong())
        for k in ("adverse_event_risk_level", "adverse_event_risk_score",
                  "adverse_event_categories", "adverse_event_penalty_hint"):
            assert k in out
        assert out["adverse_event_risk_level"] == "none"  # no flags

    def test_kosdaq_high_risk_applies_extra_penalty(self):
        row = {"ticker": "035720", "market": "KOSDAQ", "validated_score": 60}
        clean = calc_kr_segment_shadow_score(row)  # no event row
        ev = _event_row(adj=60, delta=0, flags=["⚠ 상장폐지 관련 안내"])
        adverse = calc_kr_segment_shadow_score(row, ev)
        assert adverse["adverse_event_risk_level"] == "critical"
        # severe non-dilution adverse adds a capped extra penalty (-6)
        assert adverse["kr_segment_shadow_score"] < clean["kr_segment_shadow_score"]
        assert adverse["kr_segment_shadow_delta"] == -6.0
        assert any("심각 악재" in c["label"] for c in adverse["kr_segment_shadow_components"])

    def test_kosdaq_dilution_not_double_counted(self):
        # 전환사채: existing financing flag (-5); adverse is medium → no extra for KOSDAQ
        row = {"ticker": "035720", "market": "KOSDAQ", "validated_score": 60}
        ev = _event_row(adj=50, delta=-3, flags=["⚠ 전환사채권발행결정"])
        out = calc_kr_segment_shadow_score(row, ev)
        assert out["kr_segment_shadow_delta"] == -5.0  # unchanged from financing flag only

    def test_kospi_ignores_low_confidence_watch_only(self):
        row = {"ticker": "005930", "market": "KOSPI", "validated_score": 60}
        ev = _event_row(adj=60, delta=0, flags=["👁 처분 관련"])  # watch only
        out = calc_kr_segment_shadow_score(row, ev)
        assert out["adverse_event_risk_level"] == "watch"
        assert out["kr_segment_shadow_delta"] == 0.0  # watch ignored for KOSPI
        assert out["kr_segment_shadow_score"] == 60.0

    def test_kospi_applies_medium_adverse(self):
        row = {"ticker": "005930", "market": "KOSPI", "validated_score": 60}
        ev = _event_row(adj=60, delta=0, flags=["⚠ 유상증자결정"])  # medium dilution
        out = calc_kr_segment_shadow_score(row, ev)
        assert out["adverse_event_risk_level"] == "medium"
        assert out["kr_segment_shadow_delta"] == -5.0

    def test_unknown_includes_fields_without_aggressive_change(self):
        row = {"ticker": "005930", "market": "", "validated_score": 60}
        ev = _event_row(adj=58, delta=0, flags=["⚠ 상장폐지"])
        out = calc_kr_segment_shadow_score(row, ev)
        assert out["kr_segment"] == "UNKNOWN"
        assert out["adverse_event_risk_level"] == "critical"  # field present
        assert out["kr_segment_shadow_delta"] == 0.0          # score unchanged

    def test_no_input_mutation_with_adverse(self):
        row = {"ticker": "035720", "market": "KOSDAQ", "validated_score": 60}
        ev = _event_row(adj=60, delta=0, flags=["⚠ 상장폐지"])
        rb, eb = copy.deepcopy(row), copy.deepcopy(ev)
        calc_kr_segment_shadow_score(row, ev)
        assert row == rb
        assert ev == eb

    def test_calc_output_json_serializable(self):
        ev = _event_row(adj=60, delta=0, flags=["⚠ 상장폐지", "⚠ 유상증자"])
        out = calc_kr_segment_shadow_score(
            {"ticker": "035720", "market": "KOSDAQ", "validated_score": 60}, ev)
        assert isinstance(json.dumps(out), str)

    def test_apply_includes_adverse_columns(self):
        validated_df = pd.DataFrame([
            {"ticker": "035720", "name": "C", "market": "KOSDAQ", "validated_score": 60},
        ])
        shadow_df = pd.DataFrame([
            {"ticker": "035720", "shadow_event_adjusted_score": 60, "shadow_event_delta": 0,
             "shadow_event_flags": ["⚠ 관리종목 지정"]},
        ])
        out = apply_kr_segment_shadow_composite(validated_df, shadow_df=shadow_df)
        for col in ("adverse_event_risk_level", "adverse_event_risk_score",
                    "adverse_event_categories", "adverse_event_penalty_hint"):
            assert col in out.columns
        row = out.iloc[0]
        assert row["adverse_event_risk_level"] == "high"  # 관리종목

    def test_apply_adverse_json_serializable(self):
        validated_df = pd.DataFrame([
            {"ticker": "035720", "name": "C", "market": "KOSDAQ", "validated_score": 60},
        ])
        shadow_df = pd.DataFrame([
            {"ticker": "035720", "shadow_event_adjusted_score": 60, "shadow_event_delta": 0,
             "shadow_event_flags": ["⚠ 상장폐지"]},
        ])
        out = apply_kr_segment_shadow_composite(validated_df, shadow_df=shadow_df)
        json.dumps(out.to_dict("records"))
