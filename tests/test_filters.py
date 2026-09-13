import pytest

from screener import filters


def _first_value(name):
    value = getattr(filters, name)
    return next(iter(value))


def test_us_live_etf_ticker_positive_and_negative():
    ticker = _first_value("US_LIVE_ETF_TICKERS")
    assert filters.is_us_live_etf_ticker(ticker)
    assert not filters.is_us_live_etf_ticker("NOT_A_REAL_US_TICKER")


def test_us_backtest_excluded_ticker_positive_and_negative():
    ticker = _first_value("US_BACKTEST_EXCLUDE_TICKERS")
    assert filters.is_us_backtest_excluded_ticker(ticker)
    assert not filters.is_us_backtest_excluded_ticker("NOT_A_REAL_US_TICKER")


def test_kr_backtest_excluded_ticker_positive_and_negative():
    ticker = _first_value("KR_BACKTEST_EXCLUDE_TICKERS")
    assert filters.is_kr_backtest_excluded_ticker(ticker)
    assert not filters.is_kr_backtest_excluded_ticker("000000")


def test_kr_benchmark_code_positive_and_negative():
    ticker = _first_value("KR_BENCH_CODES")
    assert filters.is_kr_benchmark_code(ticker)
    assert not filters.is_kr_benchmark_code("000000")


def test_kr_preferred_ticker_positive_and_negative():
    assert filters.is_kr_preferred_ticker("005935")
    assert not filters.is_kr_preferred_ticker("005930")
    assert not filters.is_kr_preferred_ticker("005935", exclude_preferred=False)


def test_kr_preferred_share_name_positive_and_negative():
    assert filters.is_kr_preferred_share_name("삼성전자우")
    assert filters.is_kr_preferred_share_name("현대차우선주")
    assert not filters.is_kr_preferred_share_name("삼성전자")


def test_kr_candidate_etf_name_positive_and_negative():
    keyword = _first_value("KR_CANDIDATE_ETF_NAME_KEYWORDS")
    assert filters.is_kr_candidate_etf_name(f"{keyword} 테스트")
    assert not filters.is_kr_candidate_etf_name("일반회사")


def test_kr_pipeline_etf_name_positive_and_negative():
    keyword = _first_value("KR_PIPELINE_ETF_NAME_KEYWORDS")
    assert filters.is_kr_pipeline_etf_name(f"{keyword} 테스트")
    assert not filters.is_kr_pipeline_etf_name("일반회사")


def test_kr_backtest_etf_name_positive_and_negative():
    keyword = _first_value("KR_BACKTEST_ETF_NAME_KEYWORDS")
    assert filters.is_kr_backtest_etf_name(f"{keyword} 테스트")
    assert not filters.is_kr_backtest_etf_name("일반회사")


def test_kr_precheck_excluded_security_name_positive_and_negative():
    assert filters.is_kr_precheck_excluded_security_name("테스트 ETN")
    assert filters.is_kr_precheck_excluded_security_name("테스트 스팩")
    assert filters.is_kr_precheck_excluded_security_name("테스트 리츠")
    assert not filters.is_kr_precheck_excluded_security_name("일반회사")


def test_kr_unwanted_security_positive_and_negative():
    assert filters.is_kr_unwanted_security({"name": "테스트 ETN"})
    assert filters.is_kr_unwanted_security({"name": "테스트 스팩"})
    assert not filters.is_kr_unwanted_security({"name": "일반회사"})
