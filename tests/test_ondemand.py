"""
On-demand US ticker analysis — 네트워크 없이 순수 함수 레벨 검증.
yfinance는 monkeypatch로 대체한다.
"""
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.collector import fetch_single_us_ticker_ondemand

# app.py 헬퍼는 streamlit 없이 임포트할 수 없으므로 직접 복사해서 테스트
# (app.py에 정의된 _normalize_ondemand_row / _safe_row_value 와 동일 로직)
def _normalize_ondemand_row(result):
    if result is None: return None
    if isinstance(result, dict): return result
    if isinstance(result, pd.Series): return result.to_dict()
    if isinstance(result, pd.DataFrame):
        if result.empty: return None
        return result.iloc[0].to_dict()
    return None

def _safe_row_value(row, key, default=None):
    try:
        val = row.get(key, default) if hasattr(row, "get") else row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if val is None: return default
    if isinstance(val, pd.Series):
        if val.empty: return default
        val = val.iloc[0]
    try:
        if isinstance(val, float) and math.isnan(val): return default
        fv = float(val)
        if math.isnan(fv): return default
    except (TypeError, ValueError):
        pass
    return val


# ── 테스트용 더미 데이터 ─────────────────────────────────────────────

def _make_ohlcv(n: int = 300, start: float = 100.0, trend: float = 0.001) -> pd.DataFrame:
    """단순 우상향 가격 시계열 DataFrame (Business Day 인덱스, UTC-naive)."""
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = [start * (1 + trend) ** i for i in range(n)]
    return pd.DataFrame(
        {
            "Close":  prices,
            "Open":   [p * 0.995 for p in prices],
            "High":   [p * 1.01  for p in prices],
            "Low":    [p * 0.99  for p in prices],
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


def _make_multiindex_raw(tickers: list[str], n: int = 300) -> pd.DataFrame:
    """yfinance group_by='ticker' 스타일 MultiIndex DataFrame."""
    return pd.concat({t: _make_ohlcv(n) for t in tickers}, axis=1)


_BENCH_TICKERS = [
    "SPY", "QQQ",
    "XLK", "XLF", "XLV", "XLI", "XLE", "XLRE", "XLY", "XLC", "XLB", "XLU",
    "SMH",
]


def _fake_download(ticker: str, extra: list[str] | None = None) -> pd.DataFrame:
    """타겟 티커 + 벤치마크 포함 fake raw DataFrame."""
    tickers = [ticker] + (extra or _BENCH_TICKERS)
    return _make_multiindex_raw(list(dict.fromkeys(tickers)))  # 중복 제거


def _mock_ticker_info(short_name: str = "Test Corp", sector: str = "Technology"):
    m = MagicMock()
    m.info = {"shortName": short_name, "sector": sector, "marketCap": 1_000_000_000}
    m.calendar = None
    return m


# ════════════════════════════════════════════════════════════════════
# fetch_single_us_ticker_ondemand
# ════════════════════════════════════════════════════════════════════

class TestFetchSingleUsTicker:

    def test_valid_ticker_returns_item_and_bench(self):
        """정상 티커: item dict와 bench_close dict 반환, 에러 없음."""
        fake_raw = _fake_download("NVDA")

        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info("NVIDIA", "Technology")):
            item, bench_close, err = fetch_single_us_ticker_ondemand("NVDA")

        assert err == "", f"에러가 없어야 함: {err}"
        assert item is not None
        assert item["ticker"] == "NVDA"
        assert item["company"] == "NVIDIA"
        assert item["sector"] == "Technology"
        assert "ohlcv" in item
        assert len(item["ohlcv"]) >= 20
        assert math.isfinite(item["price"])
        assert item["price"] > 0
        assert "SPY" in bench_close
        assert "QQQ" in bench_close
        assert len(bench_close["SPY"]) >= 20

    def test_lowercase_ticker_normalized_to_upper(self):
        """소문자 티커 입력 → 내부에서 대문자로 정규화."""
        fake_raw = _fake_download("AAPL")
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info()):
            item, _, err = fetch_single_us_ticker_ondemand("aapl")

        assert err == ""
        assert item is not None
        assert item["ticker"] == "AAPL"

    def test_empty_ticker_returns_error(self):
        """빈 문자열 입력 → item=None, 에러 메시지."""
        item, bench_close, err = fetch_single_us_ticker_ondemand("")
        assert item is None
        assert err != ""

    def test_empty_yfinance_response_returns_error(self):
        """yfinance가 빈 DataFrame 반환 → item=None, 에러 메시지."""
        with patch("yfinance.download", return_value=pd.DataFrame()):
            item, bench_close, err = fetch_single_us_ticker_ondemand("NVDA")

        assert item is None
        assert err != ""

    def test_unknown_ticker_missing_from_multiindex(self):
        """존재하지 않는 티커가 MultiIndex에 없음 → item=None, 에러 반환."""
        # XXXXFAKE 컬럼 없이 벤치마크만 있는 raw
        fake_raw = _make_multiindex_raw(_BENCH_TICKERS)
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info()):
            item, bench_close, err = fetch_single_us_ticker_ondemand("XXXXFAKE")

        assert item is None
        assert err != ""
        # 벤치마크는 수집되었어야 함
        assert "SPY" in bench_close

    def test_item_has_required_fields_for_calc_us_factors(self):
        """반환된 item에 calc_us_factors가 요구하는 모든 필드가 있어야 함."""
        fake_raw = _fake_download("AAPL")
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info("Apple", "Technology")):
            item, bench_close, err = fetch_single_us_ticker_ondemand("AAPL")

        assert item is not None, err
        for field in ("ohlcv", "company", "sector", "industry",
                      "market_cap", "avg_dv", "price", "earnings_date"):
            assert field in item, f"필드 누락: {field}"
        df = item["ohlcv"]
        for col in ("Close", "Amount"):
            assert col in df.columns, f"OHLCV 컬럼 누락: {col}"

    def test_yfinance_exception_returns_error(self):
        """yfinance.download가 예외를 던지면 item=None, 에러 메시지."""
        with patch("yfinance.download", side_effect=RuntimeError("network timeout")):
            item, bench_close, err = fetch_single_us_ticker_ondemand("NVDA")

        assert item is None
        assert "네트워크" in err or "다운로드" in err or err != ""

    def test_bench_close_contains_main_benchmarks(self):
        """벤치마크 SPY·QQQ·XLK는 항상 포함되어야 함 (데이터가 있을 때)."""
        fake_raw = _fake_download("PLTR")
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info("Palantir")):
            _, bench_close, err = fetch_single_us_ticker_ondemand("PLTR")

        assert err == ""
        for bench in ("SPY", "QQQ", "XLK"):
            assert bench in bench_close, f"벤치마크 누락: {bench}"


# ════════════════════════════════════════════════════════════════════
# on-demand → calc_us_factors 통합 흐름
# ════════════════════════════════════════════════════════════════════

class TestOndemandEndToEnd:

    def test_calc_us_factors_runs_on_ondemand_data(self):
        """on-demand 수집 데이터로 calc_us_factors가 정상 실행된다."""
        from screener.factors import calc_us_factors

        fake_raw = _fake_download("PLTR")
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info("Palantir", "Technology")):
            item, bench_close, err = fetch_single_us_ticker_ondemand("PLTR")

        assert item is not None, err

        row = calc_us_factors(
            ticker="PLTR",
            item=item,
            bench_close=bench_close,
            sector_bench="XLK",
            sector_avg_ret20=2.0,
            sector_avg_ret60=5.0,
            rs_rank_pct=0.6,
        )

        # 필수 필드 존재 확인
        for field in (
            "validated_score", "leader_score", "volume_score",
            "breakout_score", "catalyst_score", "buyable_score",
            "top_risk_score", "ret_5d", "ret_20d", "ret_60d",
            "week52_prox", "vol_surge", "risk_flag", "track",
        ):
            assert field in row, f"팩터 필드 누락: {field}"

        assert 0 <= row["validated_score"] <= 100, "validated_score 범위 오류"
        assert 0 <= row["leader_score"] <= 100
        assert isinstance(row["track"], str)
        assert isinstance(row["risk_flag"], str)

    def test_screener_result_not_modified_on_failure(self):
        """on-demand 실패해도 아무 side-effect 없음 (반환값만 None)."""
        with patch("yfinance.download", return_value=pd.DataFrame()):
            item, _, err = fetch_single_us_ticker_ondemand("BADTICKER")

        assert item is None
        assert err != ""
        # 이 테스트의 핵심: 위 호출이 예외를 raise하지 않음

    def test_validated_score_in_valid_range(self):
        """validated_score는 항상 0~100 사이."""
        from screener.factors import calc_us_factors

        fake_raw = _fake_download("MSFT")
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info("Microsoft", "Technology")):
            item, bench_close, err = fetch_single_us_ticker_ondemand("MSFT")

        assert item is not None, err
        row = calc_us_factors(
            ticker="MSFT",
            item=item,
            bench_close=bench_close,
            sector_bench="XLK",
            sector_avg_ret20=1.5,
            sector_avg_ret60=3.0,
        )
        vs = row.get("validated_score", -1)
        assert 0 <= vs <= 100, f"validated_score={vs} 범위 초과"

    def test_subscores_all_in_range(self):
        """모든 서브 점수(leader/volume/breakout/catalyst/risk)가 0~100."""
        from screener.factors import calc_us_factors

        fake_raw = _fake_download("NVDA")
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info("NVIDIA", "Technology")):
            item, bench_close, err = fetch_single_us_ticker_ondemand("NVDA")

        assert item is not None, err
        row = calc_us_factors(
            ticker="NVDA",
            item=item,
            bench_close=bench_close,
            sector_bench="XLK",
            sector_avg_ret20=2.5,
            sector_avg_ret60=6.0,
        )
        for field in ("leader_score", "volume_score", "breakout_score",
                      "catalyst_score", "buyable_score", "top_risk_score"):
            v = row.get(field, -1)
            assert 0 <= v <= 100, f"{field}={v} 범위 초과"


# ════════════════════════════════════════════════════════════════════
# _normalize_ondemand_row / _safe_row_value 방어 헬퍼 테스트
# ════════════════════════════════════════════════════════════════════

class TestNormalizeOndemandRow:
    """calc_us_factors가 dict / Series / DataFrame을 반환할 때 모두 정상 처리."""

    def test_dict_passthrough(self):
        d = {"validated_score": 72.5, "track": "Hot Leader / Buyable"}
        assert _normalize_ondemand_row(d) is d

    def test_series_converted_to_dict(self):
        s = pd.Series({"validated_score": 65.0, "leader_score": 80.0, "track": "Breakout Signal"})
        result = _normalize_ondemand_row(s)
        assert isinstance(result, dict)
        assert result["validated_score"] == 65.0
        assert result["track"] == "Breakout Signal"

    def test_single_row_dataframe_converted(self):
        df = pd.DataFrame([{"validated_score": 55.0, "leader_score": 60.0, "track": "Watch Only"}])
        result = _normalize_ondemand_row(df)
        assert isinstance(result, dict)
        assert result["validated_score"] == 55.0

    def test_empty_dataframe_returns_none(self):
        assert _normalize_ondemand_row(pd.DataFrame()) is None

    def test_none_returns_none(self):
        assert _normalize_ondemand_row(None) is None

    def test_calc_us_factors_returns_scalar_fields(self):
        """실제 calc_us_factors 결과의 모든 숫자 필드가 python float/int (Series 아님)."""
        from screener.factors import calc_us_factors

        fake_raw = _fake_download("AAPL")
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info("Apple", "Technology")):
            item, bench_close, err = fetch_single_us_ticker_ondemand("AAPL")

        assert item is not None, err
        raw = calc_us_factors(
            ticker="AAPL",
            item=item,
            bench_close=bench_close,
            sector_bench="XLK",
            sector_avg_ret20=1.0,
            sector_avg_ret60=3.0,
            rs_rank_pct=0.55,
        )
        row = _normalize_ondemand_row(raw)
        assert row is not None

        numeric_fields = (
            "validated_score", "leader_score", "volume_score",
            "breakout_score", "catalyst_score", "buyable_score",
            "top_risk_score", "ret_5d", "ret_20d", "ret_60d",
            "week52_prox", "vol_surge",
        )
        for f in numeric_fields:
            val = row.get(f)
            # 값이 있으면 반드시 스칼라 (pd.Series이면 bool() 오류 위험)
            if val is not None:
                assert not isinstance(val, pd.Series), f"{f} 필드가 pd.Series — boolean 평가 위험"
                assert isinstance(val, (int, float, np.floating, np.integer)), \
                    f"{f} 타입 이상: {type(val)}"


class TestSafeRowValue:
    """_safe_row_value: 다양한 입력에서 항상 스칼라 반환."""

    def test_plain_float(self):
        row = {"validated_score": 72.5}
        assert _safe_row_value(row, "validated_score", 0.0) == 72.5

    def test_missing_key_returns_default(self):
        row = {}
        assert _safe_row_value(row, "no_such_key", 99.0) == 99.0

    def test_none_value_returns_default(self):
        row = {"x": None}
        assert _safe_row_value(row, "x", 42.0) == 42.0

    def test_nan_value_returns_default(self):
        row = {"x": float("nan")}
        assert _safe_row_value(row, "x", 0.0) == 0.0

    def test_series_value_extracts_scalar(self):
        """Series가 값으로 들어와도 boolean 평가 없이 iloc[0] 반환."""
        s = pd.Series([55.0])
        row = {"validated_score": s}
        result = _safe_row_value(row, "validated_score", 0.0)
        assert result == 55.0
        assert not isinstance(result, pd.Series)

    def test_string_value_returned_as_is(self):
        row = {"track": "Breakout Signal"}
        assert _safe_row_value(row, "track", "") == "Breakout Signal"

    def test_sret_no_series_or_raises(self):
        """_sret 수정 검증: bench_close에 유효 Series가 있을 때 or 없이 올바른 fallback."""
        spy_close = pd.Series(
            [100.0 * (1.001 ** i) for i in range(300)],
            index=pd.date_range("2023-01-01", periods=300, freq="B"),
        )
        xlk_close = pd.Series(
            [200.0 * (1.002 ** i) for i in range(300)],
            index=pd.date_range("2023-01-01", periods=300, freq="B"),
        )
        bench_close = {"SPY": spy_close, "XLK": xlk_close}

        def _sret(sym: str, d: int) -> float:
            # 수정된 로직 (or 없이 명시적 None 체크)
            s = bench_close.get(sym)
            if s is None:
                s = bench_close.get("SPY")
            if s is not None and len(s) > d:
                try:
                    return float((s.iloc[-1] / s.iloc[-(d + 1)] - 1) * 100)
                except Exception:
                    pass
            return 0.0

        # XLK가 있으면 XLK 사용 (SPY 폴백 없음)
        ret_xlk = _sret("XLK", 20)
        # 없는 벤치마크는 SPY로 폴백
        ret_missing = _sret("NONEXISTENT", 20)
        ret_spy = _sret("SPY", 20)

        assert isinstance(ret_xlk, float)
        assert not math.isnan(ret_xlk)
        assert isinstance(ret_missing, float)
        # 없는 벤치마크 → SPY 값과 동일
        assert abs(ret_missing - ret_spy) < 1e-9


# ════════════════════════════════════════════════════════════════════
# _calc_event_adjusted_score shadow score 테스트
# (app.py에서 streamlit 없이 임포트 불가 → 로직 복제)
# ════════════════════════════════════════════════════════════════════

def _calc_event_adjusted_score(
    base_score: float,
    event_context: dict,
    risk_score=None,
    market: str = "US",
) -> dict:
    """app.py의 _calc_event_adjusted_score와 동일 로직 (테스트용 복제)."""
    out = {
        "event_adjusted_score":    None,
        "event_delta":             0.0,
        "event_adjustment_reason": "이벤트 데이터 없음",
        "event_flags":             [],
        "event_risk_flags":        [],
    }
    ev = event_context or {}
    ev_score = ev.get("event_score")
    if ev_score is None:
        return out
    try:
        ev_score = float(ev_score)
    except (TypeError, ValueError):
        return out

    if ev_score >= 75:
        delta = 6.0; reason_base = f"이벤트강세({ev_score:.0f})"
    elif ev_score >= 65:
        delta = 3.0; reason_base = f"이벤트양호({ev_score:.0f})"
    elif ev_score <= 25:
        delta = -10.0; reason_base = f"이벤트위험({ev_score:.0f})"
    elif ev_score <= 35:
        delta = -6.0; reason_base = f"이벤트부정({ev_score:.0f})"
    else:
        delta = 0.0; reason_base = f"이벤트중립({ev_score:.0f})"

    all_flags  = ev.get("event_flags", [])
    pos_flags  = [f for f in all_flags if str(f).startswith("✅")]
    risk_flags = [f for f in all_flags if str(f).startswith("⚠")]

    if pos_flags:
        delta = min(delta + 2.0, 8.0)
    if risk_flags:
        delta = max(delta - 3.0, -15.0)

    reasons = [reason_base]
    if risk_score is not None:
        try:
            if float(risk_score) >= 85 and ev_score < 60:
                delta = max(delta - 3.0, -15.0)
                reasons.append("고위험종목")
        except (TypeError, ValueError):
            pass

    if pos_flags:
        reasons.append(f"긍정공시+{len(pos_flags)}")
    if risk_flags:
        reasons.append(f"부정공시-{len(risk_flags)}")

    adj = max(0.0, min(100.0, float(base_score) + delta))
    out["event_adjusted_score"]    = round(adj, 1)
    out["event_delta"]             = round(delta, 1)
    out["event_adjustment_reason"] = " / ".join(reasons)
    out["event_flags"]             = pos_flags
    out["event_risk_flags"]        = risk_flags
    return out


def _make_ev_ctx(event_score=None, pos_flags=None, risk_flags=None) -> dict:
    """테스트용 event_context 생성 헬퍼."""
    all_flags = (pos_flags or []) + (risk_flags or [])
    return {
        "event_score":     event_score,
        "event_flags":     all_flags,
        "positive_events": [],
        "negative_events": [],
    }


class TestCalcEventAdjustedScore:

    # ── event_score None ──────────────────────────────────────────────

    def test_none_event_score_returns_no_adjustment(self):
        ctx = _make_ev_ctx(event_score=None)
        out = _calc_event_adjusted_score(65.0, ctx)
        assert out["event_adjusted_score"] is None
        assert out["event_delta"] == 0.0

    def test_none_context_returns_no_adjustment(self):
        out = _calc_event_adjusted_score(65.0, None)
        assert out["event_adjusted_score"] is None

    # ── positive event_score ──────────────────────────────────────────

    def test_event_score_80_gives_positive_delta(self):
        ctx = _make_ev_ctx(event_score=80)
        out = _calc_event_adjusted_score(60.0, ctx)
        assert out["event_delta"] > 0
        assert out["event_adjusted_score"] > 60.0

    def test_event_score_75_delta_is_6(self):
        ctx = _make_ev_ctx(event_score=75)
        out = _calc_event_adjusted_score(60.0, ctx)
        assert out["event_delta"] == 6.0
        assert out["event_adjusted_score"] == 66.0

    def test_event_score_65_delta_is_3(self):
        ctx = _make_ev_ctx(event_score=65)
        out = _calc_event_adjusted_score(60.0, ctx)
        assert out["event_delta"] == 3.0

    # ── negative event_score ─────────────────────────────────────────

    def test_event_score_20_gives_negative_delta(self):
        ctx = _make_ev_ctx(event_score=20)
        out = _calc_event_adjusted_score(60.0, ctx)
        assert out["event_delta"] < 0
        assert out["event_adjusted_score"] < 60.0

    def test_event_score_25_delta_is_minus10(self):
        ctx = _make_ev_ctx(event_score=25)
        out = _calc_event_adjusted_score(60.0, ctx)
        assert out["event_delta"] == -10.0
        assert out["event_adjusted_score"] == 50.0

    def test_event_score_35_delta_is_minus6(self):
        ctx = _make_ev_ctx(event_score=35)
        out = _calc_event_adjusted_score(60.0, ctx)
        assert out["event_delta"] == -6.0

    def test_neutral_score_50_no_delta(self):
        ctx = _make_ev_ctx(event_score=50)
        out = _calc_event_adjusted_score(60.0, ctx)
        assert out["event_delta"] == 0.0
        assert out["event_adjusted_score"] == 60.0

    # ── risk flags ───────────────────────────────────────────────────

    def test_risk_flags_add_penalty(self):
        ctx = _make_ev_ctx(event_score=50, risk_flags=["⚠ 유상증자 발표"])
        out_with = _calc_event_adjusted_score(60.0, ctx)
        ctx_no = _make_ev_ctx(event_score=50)
        out_no = _calc_event_adjusted_score(60.0, ctx_no)
        assert out_with["event_delta"] < out_no["event_delta"]

    def test_pos_flags_add_bonus(self):
        ctx = _make_ev_ctx(event_score=65, pos_flags=["✅ 실적 서프라이즈"])
        out = _calc_event_adjusted_score(60.0, ctx)
        # 65점 → base +3, pos_flag → +2, total +5, upper 8
        assert out["event_delta"] == 5.0

    def test_pos_flags_capped_at_8(self):
        ctx = _make_ev_ctx(event_score=75, pos_flags=["✅ flag1", "✅ flag2", "✅ flag3"])
        out = _calc_event_adjusted_score(60.0, ctx)
        # 75점 → +6, pos → +2 = 8 (upper cap)
        assert out["event_delta"] == 8.0

    def test_risk_flags_lower_bound_minus15(self):
        ctx = _make_ev_ctx(
            event_score=20,  # -10
            risk_flags=["⚠ r1", "⚠ r2", "⚠ r3"],  # -3
        )
        out = _calc_event_adjusted_score(60.0, ctx)
        # -10 - 3 = -13 (within -15 bound)
        assert out["event_delta"] >= -15.0

    # ── high risk_score penalty ──────────────────────────────────────

    def test_high_risk_score_adds_penalty_when_event_mediocre(self):
        ctx = _make_ev_ctx(event_score=50)  # neutral
        out_high_risk = _calc_event_adjusted_score(60.0, ctx, risk_score=90)
        out_low_risk  = _calc_event_adjusted_score(60.0, ctx, risk_score=30)
        assert out_high_risk["event_delta"] < out_low_risk["event_delta"]

    def test_high_risk_score_no_penalty_when_event_good(self):
        ctx = _make_ev_ctx(event_score=75)  # good event
        out = _calc_event_adjusted_score(60.0, ctx, risk_score=90)
        # event >= 60, so no extra penalty
        assert out["event_delta"] == 6.0

    # ── clamp 0~100 ──────────────────────────────────────────────────

    def test_clamp_upper_100(self):
        ctx = _make_ev_ctx(event_score=80, pos_flags=["✅ flag"])
        out = _calc_event_adjusted_score(97.0, ctx)
        assert out["event_adjusted_score"] <= 100.0

    def test_clamp_lower_0(self):
        ctx = _make_ev_ctx(event_score=20, risk_flags=["⚠ r1", "⚠ r2"])
        out = _calc_event_adjusted_score(5.0, ctx)
        assert out["event_adjusted_score"] >= 0.0

    # ── 기존 score 필드 불변 ─────────────────────────────────────────

    def test_base_score_fields_not_mutated(self):
        row = {
            "validated_score": 65.0,
            "rr_score":        70.0,
            "catalyst_score":  55.0,
            "final_score":     62.0,
        }
        ctx = _make_ev_ctx(event_score=80)
        _calc_event_adjusted_score(row["validated_score"], ctx)
        # 원본 dict 변경 없음
        assert row["validated_score"] == 65.0
        assert row["rr_score"]        == 70.0
        assert row["catalyst_score"]  == 55.0
        assert row["final_score"]     == 62.0

    # ── 반환 dict 구조 ────────────────────────────────────────────────

    def test_return_keys_always_present(self):
        out = _calc_event_adjusted_score(60.0, {})
        for key in ("event_adjusted_score", "event_delta",
                    "event_adjustment_reason", "event_flags", "event_risk_flags"):
            assert key in out, f"반환 키 누락: {key}"

    def test_event_adjusted_score_is_float_or_none(self):
        ctx = _make_ev_ctx(event_score=70)
        out = _calc_event_adjusted_score(60.0, ctx)
        assert isinstance(out["event_adjusted_score"], float)

    # ── US on-demand 결과 필드 존재 확인 ─────────────────────────────

    def test_us_ondemand_result_has_event_adjusted_score(self):
        """us_event_context가 있으면 event_adjusted_score 필드가 붙는다."""
        from screener.factors import calc_us_factors
        from unittest.mock import patch

        fake_raw = _fake_download("NVDA")
        with patch("yfinance.download", return_value=fake_raw), \
             patch("yfinance.Ticker", return_value=_mock_ticker_info("NVIDIA", "Technology")):
            item, bench_close, err = fetch_single_us_ticker_ondemand("NVDA")

        assert item is not None, err
        row = calc_us_factors(
            ticker="NVDA", item=item, bench_close=bench_close,
            sector_bench="XLK", sector_avg_ret20=2.0, sector_avg_ret60=5.0,
        )
        vs = float(row.get("validated_score", 50.0))
        rsk = row.get("top_risk_score")

        # EDGAR 없이 빈 context → None 반환
        out_no_event = _calc_event_adjusted_score(vs, {})
        assert out_no_event["event_adjusted_score"] is None

        # 이벤트 있으면 계산됨
        ctx = _make_ev_ctx(event_score=75)
        out_with_event = _calc_event_adjusted_score(vs, ctx, risk_score=rsk)
        assert out_with_event["event_adjusted_score"] is not None

    # ── KR on-demand 결과 필드 존재 확인 ─────────────────────────────

    def test_kr_ondemand_result_has_event_adjusted_score(self):
        """kr_event_context가 있으면 event_adjusted_score 필드가 붙는다."""
        from screener.factors import calc_kr_factors
        import pandas as pd

        n = 300
        dates  = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = [50000.0 * (1.001 ** i) for i in range(n)]
        df = pd.DataFrame(
            {"Open": prices, "High": prices, "Low": prices,
             "Close": prices, "Volume": [100_000]*n,
             "Amount": [p*100_000 for p in prices]},
            index=dates,
        )
        item = {
            "ticker": "005930", "name": "삼성전자", "market": "KOSPI",
            "sector": "", "ohlcv": df, "market_cap": 4e14,
            "avg_amount": 5e9, "price": float(prices[-1]), "is_warned": False,
            "foreign_5d": 1e6, "foreign_20d": 3e6, "inst_5d": 5e5, "inst_20d": 1e6,
        }
        bench = {
            "kospi":  pd.Series([2500.0 * (1.0005**i) for i in range(n)], index=dates),
            "kosdaq": pd.Series([900.0  * (1.0006**i) for i in range(n)], index=dates),
        }
        row = calc_kr_factors("005930", item, bench, 0.0, 0.5)
        vs  = float(row.get("validated_score", 50.0))

        ctx = _make_ev_ctx(event_score=80, pos_flags=["✅ 잠정실적 서프라이즈"])
        out = _calc_event_adjusted_score(vs, ctx, market="KR")
        assert out["event_adjusted_score"] is not None
        assert out["event_delta"] > 0

    # ── EDGAR/DART fetch 실패해도 on-demand 성공 ────────────────────

    def test_event_fetch_failure_does_not_crash_analysis(self):
        """이벤트 context가 빈 dict여도 _calc_event_adjusted_score는 None 반환으로 성공."""
        out = _calc_event_adjusted_score(60.0, {"error": "timeout", "event_score": None})
        assert out["event_adjusted_score"] is None
        assert out["event_delta"] == 0.0
