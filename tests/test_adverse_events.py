"""
screener/adverse_events.py — KR Adverse Event Risk Classifier v2 tests.
Pure / offline. No network.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.adverse_events import classify_kr_adverse_event_risk


def _c(flags=None, names=None, events=None):
    return classify_kr_adverse_event_risk(event_flags=flags, event_names=names, events=events)


class TestAdverseClassifier:

    # ── dilution / financing ─────────────────────────────────────────

    def test_dilution_financing_detection(self):
        out = _c(["⚠ 주요사항보고서(유상증자결정)"])
        assert "dilution_financing" in out["adverse_event_categories"]
        assert out["adverse_event_risk_level"] == "medium"
        assert out["adverse_event_penalty_hint"] == -5

    def test_convertible_bond_korean(self):
        out = _c(["전환사채권발행결정"])
        assert "dilution_financing" in out["adverse_event_categories"]
        assert out["adverse_event_risk_level"] == "medium"

    def test_cb_bw_eb_abbreviations(self):
        for kw in ("CB issuance", "BW offering", "EB notes"):
            out = _c([kw])
            assert "dilution_financing" in out["adverse_event_categories"]
            assert out["adverse_event_risk_level"] == "high"
            assert out["adverse_event_penalty_hint"] == -8

    def test_refixing_is_high(self):
        out = _c(["전환가액 리픽싱 조정"])
        assert out["adverse_event_risk_level"] == "high"

    def test_large_offering_escalates_to_high(self):
        out = _c(["대규모 유상증자 결정"])
        assert out["adverse_event_risk_level"] == "high"

    # ── ownership / control ──────────────────────────────────────────

    def test_ownership_negative_with_context_is_medium(self):
        out = _c(["주식등의대량보유상황보고서(처분)"])
        assert "ownership_control" in out["adverse_event_categories"]
        assert out["adverse_event_risk_level"] == "medium"

    def test_ownership_watch_does_not_overpenalize(self):
        # bare disposal word, no stake context → watch, low confidence, small penalty
        out = _c(["처분"])
        assert out["adverse_event_risk_level"] == "watch"
        assert out["adverse_event_confidence"] == "low"
        assert out["adverse_event_penalty_hint"] == -2

    def test_largest_shareholder_change_is_high(self):
        out = _c(["최대주주변경"])
        assert out["adverse_event_risk_level"] == "high"
        assert "ownership_control" in out["adverse_event_categories"]

    # ── listing / accounting / regulatory ────────────────────────────

    def test_listing_critical_risk(self):
        for kw in ("상장폐지", "거래정지", "의견거절", "횡령", "배임"):
            out = _c([kw])
            assert out["adverse_event_risk_level"] == "critical", kw
            assert out["adverse_event_risk_score"] >= 90
            assert out["adverse_event_penalty_hint"] == -12

    def test_management_issue_is_high(self):
        out = _c(["관리종목 지정"])
        assert out["adverse_event_risk_level"] == "high"

    def test_investment_caution_is_watch(self):
        out = _c(["투자주의 종목"])
        assert out["adverse_event_risk_level"] == "watch"

    def test_audit_opinion_alone_is_watch(self):
        out = _c(["감사의견 관련 공시"])
        assert out["adverse_event_risk_level"] == "watch"

    def test_audit_disclaimer_is_critical(self):
        out = _c(["감사의견 의견거절"])
        assert out["adverse_event_risk_level"] == "critical"

    # ── business stress ──────────────────────────────────────────────

    def test_business_stress_litigation(self):
        out = _c(["소송제기 공시"])
        assert "business_stress" in out["adverse_event_categories"]
        assert out["adverse_event_risk_level"] == "medium"

    def test_business_stress_rehabilitation_is_critical(self):
        out = _c(["회생절차 개시신청"])
        assert out["adverse_event_risk_level"] == "critical"

    def test_contract_termination_medium(self):
        out = _c(["주요계약해지 공시"])
        assert out["adverse_event_risk_level"] == "medium"

    # ── aggregation behavior ─────────────────────────────────────────

    def test_multiple_categories_increase_score(self):
        single = _c(["유상증자"])
        multi  = _c(["유상증자", "소송제기"])  # dilution + business, both medium
        assert multi["adverse_event_risk_score"] > single["adverse_event_risk_score"]
        assert len(multi["adverse_event_categories"]) == 2

    def test_critical_dominates(self):
        out = _c(["유상증자", "상장폐지"])  # medium + critical
        assert out["adverse_event_risk_level"] == "critical"
        assert out["adverse_event_risk_score"] >= 90

    # ── edge cases ───────────────────────────────────────────────────

    def test_empty_input_returns_none(self):
        out = _c([])
        assert out["adverse_event_risk_level"] == "none"
        assert out["adverse_event_risk_score"] == 0
        assert out["adverse_event_penalty_hint"] == 0
        assert out["adverse_event_categories"] == []

    def test_all_none_input(self):
        out = classify_kr_adverse_event_risk()
        assert out["adverse_event_risk_level"] == "none"

    def test_malformed_input_does_not_crash(self):
        assert _c(12345)["adverse_event_risk_level"] == "none"
        assert _c({"weird": "dict"})["adverse_event_risk_level"] == "none"
        assert _c([None, 1, {"report_nm": "상장폐지"}])["adverse_event_risk_level"] == "critical"

    def test_string_flag_input(self):
        out = _c("유상증자결정")  # plain string, not list
        assert out["adverse_event_risk_level"] == "medium"

    def test_events_dicts_input(self):
        events = [{"report_nm": "전환사채권발행결정"}, {"report_nm": "사업보고서"}]
        out = classify_kr_adverse_event_risk(events=events)
        assert out["adverse_event_risk_level"] == "medium"

    def test_event_names_input(self):
        out = classify_kr_adverse_event_risk(event_names=["관리종목지정"])
        assert out["adverse_event_risk_level"] == "high"

    def test_space_normalization(self):
        out = _c(["최 대 주 주 변 경"])  # spaces inside keyword
        assert out["adverse_event_risk_level"] == "high"

    def test_output_keys_and_types(self):
        out = _c(["상장폐지"])
        assert set(out.keys()) == {
            "adverse_event_risk_level", "adverse_event_risk_score",
            "adverse_event_categories", "adverse_event_reasons",
            "adverse_event_penalty_hint", "adverse_event_confidence",
        }
        assert isinstance(out["adverse_event_risk_score"], int)
        assert isinstance(out["adverse_event_categories"], list)
        assert isinstance(out["adverse_event_reasons"], list)

    def test_json_serializable(self):
        out = _c(["상장폐지", "유상증자", "소송"])
        assert isinstance(json.dumps(out), str)

    def test_score_clamped_0_100(self):
        out = _c(["상장폐지", "거래정지", "횡령", "배임", "회생절차",
                  "유상증자", "소송", "최대주주변경", "관리종목"])
        assert 0 <= out["adverse_event_risk_score"] <= 100
