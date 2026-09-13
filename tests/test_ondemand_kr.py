"""
On-demand KR ticker analysis — 네트워크 없이 순수 함수 레벨 검증.
Naver Finance / fetch_kr_benchmarks 는 monkeypatch로 대체한다.
"""
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.collector import (
    _build_kr_offline_name_map,
    fetch_single_kr_ticker_ondemand,
)

# ─ 헬퍼 함수 복사 (app.py 를 streamlit 없이 임포트하기 위해) ────────────


def _normalize_ondemand_row(result):
    if result is None:
        return None
    if isinstance(result, dict):
        return result
    if isinstance(result, pd.Series):
        return result.to_dict()
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return None
        return result.iloc[0].to_dict()
    return None


def _safe_row_value(row, key: str, default=None):
    try:
        val = row.get(key, default) if hasattr(row, "get") else row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if val is None:
        return default
    if isinstance(val, pd.Series):
        if val.empty:
            return default
        val = val.iloc[0]
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
        fv = float(val)
        if math.isnan(fv):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _is_kr_query(query: str) -> bool:
    import re
    q = str(query).strip()
    if re.match(r"^\d{6}$", q):
        return True
    if re.search(r"[가-힣]", q):
        return True
    return False


# ─ 더미 데이터 팩토리 ────────────────────────────────────────────────

def _make_ohlcv_kr(n: int = 300, start: float = 50000.0, trend: float = 0.001) -> pd.DataFrame:
    """KR 스타일 OHLCV DataFrame (원화 가격, Business Day 인덱스)."""
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = [start * (1 + trend) ** i for i in range(n)]
    vol    = [100_000 + i * 100 for i in range(n)]
    return pd.DataFrame(
        {
            "Open":   [p * 0.99 for p in prices],
            "High":   [p * 1.01 for p in prices],
            "Low":    [p * 0.98 for p in prices],
            "Close":  prices,
            "Volume": vol,
            "Amount": [p * v for p, v in zip(prices, vol)],
        },
        index=dates,
    )


def _make_kr_bench(n: int = 300, base: float = 2500.0) -> dict[str, pd.Series]:
    """KOSPI / KOSDAQ 벤치마크 Series."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return {
        "kospi":  pd.Series([base * (1.0005 ** i) for i in range(n)], index=dates),
        "kosdaq": pd.Series([base * 0.4 * (1.0006 ** i) for i in range(n)], index=dates),
    }


def _fake_item(code: str = "005930", name: str = "삼성전자", market: str = "KOSPI") -> dict:
    df = _make_ohlcv_kr()
    return {
        "ticker":     code,
        "name":       name,
        "market":     market,
        "sector":     "",
        "ohlcv":      df,
        "market_cap": 400_000_000_000_000,  # 400조 원
        "avg_amount": float(df["Amount"].tail(20).mean()),
        "price":      float(df["Close"].iloc[-1]),
        "is_warned":  False,
        "foreign_5d":  1_000_000.0,
        "foreign_20d": 3_000_000.0,
        "inst_5d":     500_000.0,
        "inst_20d":    1_000_000.0,
    }


# ════════════════════════════════════════════════════════════════════
# _build_kr_offline_name_map 테스트
# ════════════════════════════════════════════════════════════════════


class TestBuildKrOfflineNameMap:

    def test_contains_samsung(self):
        m = _build_kr_offline_name_map()
        assert "삼성전자" in m
        assert m["삼성전자"] == "005930"

    def test_contains_sk_hynix(self):
        m = _build_kr_offline_name_map()
        assert "SK하이닉스" in m
        assert m["SK하이닉스"] == "000660"

    def test_returns_dict(self):
        m = _build_kr_offline_name_map()
        assert isinstance(m, dict)
        assert len(m) >= 2

    def test_values_are_6digit_codes(self):
        import re
        m = _build_kr_offline_name_map()
        for name, code in list(m.items())[:30]:
            assert re.match(r"^\d{6}$", code), f"코드 형식 오류: {name} → {code}"


# ════════════════════════════════════════════════════════════════════
# fetch_single_kr_ticker_ondemand 수집 함수 테스트
# ════════════════════════════════════════════════════════════════════


class TestFetchSingleKrTicker:

    def _mock_naver_kr_meta(self, code: str):
        return {"005930": ("삼성전자", "KOSPI"), "000660": ("SK하이닉스", "KOSPI")}.get(
            code, (code, "KOSPI")
        )

    def test_empty_query_returns_error(self):
        item, _, err = fetch_single_kr_ticker_ondemand("")
        assert item is None
        assert err != ""

    def test_6digit_code_triggers_naver_lookup(self):
        with patch("screener.collector._naver_kr_meta", return_value=("삼성전자", "KOSPI")), \
             patch("screener.collector._fetch_kr_one", return_value=_fake_item()), \
             patch("screener.collector.fetch_kr_benchmarks", return_value=_make_kr_bench()):
            item, bench, err = fetch_single_kr_ticker_ondemand("005930")
        assert item is not None, err
        assert err == ""
        assert item["ticker"] == "005930"

    def test_known_name_resolves_to_code(self):
        with patch("screener.collector._naver_kr_meta", return_value=("삼성전자", "KOSPI")), \
             patch("screener.collector._fetch_kr_one", return_value=_fake_item()), \
             patch("screener.collector.fetch_kr_benchmarks", return_value=_make_kr_bench()):
            item, bench, err = fetch_single_kr_ticker_ondemand("삼성전자")
        assert item is not None, err
        assert item["ticker"] == "005930"

    def test_unknown_name_returns_error(self):
        item, _, err = fetch_single_kr_ticker_ondemand("존재하지않는종목XYZ")
        assert item is None
        assert "인식할 수 없습니다" in err

    def test_naver_ohlcv_failure_returns_error(self):
        with patch("screener.collector._naver_kr_meta", return_value=("삼성전자", "KOSPI")), \
             patch("screener.collector._fetch_kr_one", return_value=None), \
             patch("screener.collector.fetch_kr_benchmarks", return_value=_make_kr_bench()):
            item, _, err = fetch_single_kr_ticker_ondemand("005930")
        assert item is None
        assert err != ""

    def test_insufficient_ohlcv_returns_error(self):
        thin_item = _fake_item()
        thin_item["ohlcv"] = _make_ohlcv_kr(n=30)  # 30일 < 60일
        with patch("screener.collector._naver_kr_meta", return_value=("삼성전자", "KOSPI")), \
             patch("screener.collector._fetch_kr_one", return_value=thin_item), \
             patch("screener.collector.fetch_kr_benchmarks", return_value=_make_kr_bench()):
            item, _, err = fetch_single_kr_ticker_ondemand("005930")
        assert item is None
        assert "부족" in err

    def test_bench_returned_has_kospi_and_kosdaq(self):
        with patch("screener.collector._naver_kr_meta", return_value=("삼성전자", "KOSPI")), \
             patch("screener.collector._fetch_kr_one", return_value=_fake_item()), \
             patch("screener.collector.fetch_kr_benchmarks", return_value=_make_kr_bench()):
            _, bench, err = fetch_single_kr_ticker_ondemand("005930")
        assert err == ""
        assert "kospi" in bench
        assert "kosdaq" in bench

    def test_item_has_required_fields(self):
        with patch("screener.collector._naver_kr_meta", return_value=("삼성전자", "KOSPI")), \
             patch("screener.collector._fetch_kr_one", return_value=_fake_item()), \
             patch("screener.collector.fetch_kr_benchmarks", return_value=_make_kr_bench()):
            item, _, err = fetch_single_kr_ticker_ondemand("005930")
        assert err == ""
        for field in ("ticker", "name", "market", "ohlcv", "market_cap",
                      "avg_amount", "price", "foreign_5d", "foreign_20d",
                      "inst_5d", "inst_20d"):
            assert field in item, f"item 필드 누락: {field}"


# ════════════════════════════════════════════════════════════════════
# calc_kr_factors 통합 흐름
# ════════════════════════════════════════════════════════════════════


class TestKrOndemandEndToEnd:

    def test_calc_kr_factors_runs_on_ondemand_data(self):
        """on-demand 수집 데이터로 calc_kr_factors가 정상 실행된다."""
        from screener.factors import calc_kr_factors

        item  = _fake_item("005930", "삼성전자", "KOSPI")
        bench = _make_kr_bench()

        row = calc_kr_factors(
            ticker="005930",
            item=item,
            bench_close=bench,
            sector_avg_ret20=1.5,
            rs_rank_pct=0.6,
        )

        for field in (
            "validated_score", "leader_score", "volume_score",
            "breakout_score", "catalyst_score", "buyable_score",
            "top_risk_score", "supply_score", "ret_5d", "ret_20d", "ret_60d",
            "week52_prox", "vol_surge", "risk_flag", "track",
        ):
            assert field in row, f"팩터 필드 누락: {field}"

        assert 0 <= row["validated_score"] <= 100
        assert 0 <= row["supply_score"] <= 100
        assert isinstance(row["track"], str)
        assert isinstance(row["risk_flag"], str)

    def test_calc_kr_factors_scalar_fields(self):
        """calc_kr_factors 결과의 숫자 필드가 모두 스칼라."""
        from screener.factors import calc_kr_factors

        item  = _fake_item("000660", "SK하이닉스", "KOSPI")
        bench = _make_kr_bench()
        raw   = calc_kr_factors("000660", item, bench, 2.0, 0.55)
        row   = _normalize_ondemand_row(raw)
        assert row is not None

        numeric_fields = (
            "validated_score", "leader_score", "volume_score",
            "breakout_score", "catalyst_score", "supply_score",
            "top_risk_score", "ret_5d", "ret_20d", "ret_60d",
        )
        for f in numeric_fields:
            val = row.get(f)
            if val is not None:
                assert not isinstance(val, pd.Series), f"{f} 필드가 pd.Series"
                assert isinstance(val, (int, float, np.floating, np.integer)), \
                    f"{f} 타입 이상: {type(val)}"

    def test_dataframe_result_normalized(self):
        """calc_kr_factors 결과가 DataFrame으로 오면 _normalize_ondemand_row가 처리."""
        from screener.factors import calc_kr_factors

        item  = _fake_item()
        bench = _make_kr_bench()
        raw   = calc_kr_factors("005930", item, bench, 0.0, 0.5)
        # dict 이지만 일부러 DataFrame으로 감싸서 테스트
        df = pd.DataFrame([raw])
        result = _normalize_ondemand_row(df)
        assert isinstance(result, dict)
        assert "validated_score" in result

    def test_failure_no_side_effects(self):
        """fetch_single_kr_ticker_ondemand 실패해도 예외 없이 error_msg만 반환."""
        with patch("screener.collector._naver_kr_meta", return_value=("삼성전자", "KOSPI")), \
             patch("screener.collector._fetch_kr_one", return_value=None):
            item, _, err = fetch_single_kr_ticker_ondemand("005930")
        assert item is None
        assert err != ""

    def test_supply_fields_missing_safe(self):
        """supply_score/supply_flow 필드 없어도 _safe_row_value가 None 반환."""
        row = {"validated_score": 60.0, "ret_20d": 3.0}
        assert _safe_row_value(row, "supply_score") is None
        assert _safe_row_value(row, "supply_flow") is None
        # 기본값 지정 시 그 값 반환
        assert _safe_row_value(row, "supply_score", 50.0) == 50.0

    def test_kosdaq_bench_key_resolved(self):
        """KOSDAQ 종목이면 kosdaq 벤치마크를 올바르게 선택한다."""
        item  = _fake_item("086520", "에코프로", "KOSDAQ")
        bench = _make_kr_bench()
        assert item["market"] == "KOSDAQ"
        mkt_key = "kospi" if item.get("market", "KOSPI") == "KOSPI" else "kosdaq"
        assert mkt_key == "kosdaq"
        assert mkt_key in bench


# ════════════════════════════════════════════════════════════════════
# _is_kr_query 쿼리 패턴 판별
# ════════════════════════════════════════════════════════════════════


class TestIsKrQuery:

    def test_6digit_code_is_kr(self):
        assert _is_kr_query("005930") is True
        assert _is_kr_query("000660") is True
        assert _is_kr_query("086520") is True

    def test_us_ticker_is_not_kr(self):
        assert _is_kr_query("NVDA") is False
        assert _is_kr_query("AAPL") is False
        assert _is_kr_query("PLTR") is False

    def test_korean_name_is_kr(self):
        assert _is_kr_query("삼성전자") is True
        assert _is_kr_query("에코프로") is True
        assert _is_kr_query("SK하이닉스") is True  # 한글 포함

    def test_empty_string_is_not_kr(self):
        assert _is_kr_query("") is False

    def test_5digit_number_is_not_kr(self):
        # 5자리 숫자는 KR 코드 아님
        assert _is_kr_query("12345") is False

    def test_7digit_number_is_not_kr(self):
        assert _is_kr_query("1234567") is False
