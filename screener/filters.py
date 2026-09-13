"""
Pure ticker/name filters shared by the screener, UI, and backtests.

Keep market-specific keyword sets separate: live KR screening, UI candidate
filtering, and KR backtest prechecks currently use different lists.
"""

import re

from screener.universe import UNIVERSE_ETFS, US_BENCH


US_LIVE_ETF_TICKERS = (
    set(US_BENCH.values()) | set(UNIVERSE_ETFS) | {"XLC", "XLY", "XLP", "XLB", "XLRE"}
)

US_BACKTEST_EXCLUDE_TICKERS = {
    # 시장 벤치마크
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV", "MDY", "IJR", "EFA", "EEM",
    # 섹터 ETF
    "SMH", "SOXX", "XLK", "IGV", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI",
    "XLE", "XLB", "XLU", "XLRE", "CIBR", "BUG", "ITA", "XAR", "IBB", "XBI",
    "GRID", "AMPS", "WCLD", "GDX", "SLV", "GLD", "TLT", "HYG", "LQD",
    # 레버리지/인버스
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UVXY", "VXX",
}

KR_CANDIDATE_ETF_NAME_KEYWORDS = {
    "KODEX", "TIGER", "KBSTAR", "HANARO", "ARIRANG", "KOSEF",
    "ACE", "TIMEFOLIO", "KINDEX", "TREX", "SOL", "ETF", "ETN",
    "액티브", "인버스", "레버리지", "글로벌AI", "인덱스",
    "퀀텀", "스마트", "TDF", "리츠", "채권", "국채",
}

KR_PIPELINE_ETF_NAME_KEYWORDS = {
    "KODEX", "TIGER", "KBSTAR", "HANARO", "ARIRANG", "KOSEF", "ACE",
    "TIMEFOLIO", "KINDEX", "TREX", "SOL", "마이티", "파워", "PLUS", "네비게이터",
    "액티브", "인버스", "레버리지", "ETN", "ETF", "글로벌AI", "글로벌", "인덱스",
    "퀀텀", "스마트", "하이일드", "TDF", "리츠", "MMF", "채권", "국채", "금리",
}

KR_BACKTEST_EXCLUDE_TICKERS = {
    # KODEX 시리즈 (삼성자산운용)
    "069500", "122630", "229200", "251340", "252670", "114800", "233740",
    "091160", "091170", "152100", "183700", "278530", "261220", "267490",
    "244620", "292190", "305080", "395160", "441800", "445290", "453810",
    # TIGER 시리즈 (미래에셋)
    "102110", "143460", "157490", "195930", "214980", "227560", "261110",
    "278540", "305540", "364980", "381170", "394670", "411060", "441760",
    # KBSTAR / HANARO / ARIRANG 등
    "211560", "287300", "292150", "315960", "361580", "396500", "426410",
    # 이름 기반 (수집 시 이름으로도 필터)
    "kodex", "tiger", "kbstar", "hanaro", "arirang", "smart", "kosef",
    "ace_etf", "timefolio", "kindex", "trex", "kb_etf",
}

KR_BACKTEST_ETF_NAME_KEYWORDS = {
    "KODEX", "TIGER", "KBSTAR", "HANARO", "ARIRANG", "KOSEF", "ACE",
    "TIMEFOLIO", "KINDEX", "TREX", "SOL ETF", "SMART", "ETF",
}

KR_BENCH_CODES = {"069500", "122630", "kospi", "kosdaq"}

KR_BACKTEST_BAD_SECURITY_KEYWORDS = {
    "ETF", "ETN", "스팩", "SPAC", "리츠", "REIT", "인버스", "레버리지",
}

KR_PRECHECK_BAD_SECURITY_KEYWORDS = {"ETN", "스팩", "SPAC", "리츠", "REIT"}
KR_PREFERRED_SHARE_NAME_RE = r"(\d?우[A-Z]?$|우선주$)"


def contains_any_keyword(text: str, keywords) -> bool:
    upper = str(text or "").upper()
    return any(kw in upper for kw in keywords)


def is_us_live_etf_ticker(ticker: str) -> bool:
    return ticker in US_LIVE_ETF_TICKERS


def is_us_backtest_excluded_ticker(ticker: str) -> bool:
    return ticker in US_BACKTEST_EXCLUDE_TICKERS


def is_kr_backtest_excluded_ticker(ticker: str) -> bool:
    return ticker in KR_BACKTEST_EXCLUDE_TICKERS


def is_kr_benchmark_code(ticker: str) -> bool:
    return ticker in KR_BENCH_CODES


def is_kr_preferred_ticker(ticker: str, exclude_preferred: bool = True) -> bool:
    return (
        exclude_preferred
        and isinstance(ticker, str)
        and len(ticker) == 6
        and ticker.isdigit()
        and ticker[-1] != "0"
    )


def is_kr_preferred_share_name(name: str) -> bool:
    return bool(re.search(KR_PREFERRED_SHARE_NAME_RE, str(name or "").strip()))


def is_kr_candidate_etf_name(name: str) -> bool:
    return contains_any_keyword(name, KR_CANDIDATE_ETF_NAME_KEYWORDS)


def is_kr_pipeline_etf_name(name: str) -> bool:
    return contains_any_keyword(name, KR_PIPELINE_ETF_NAME_KEYWORDS)


def is_kr_backtest_etf_name(name: str) -> bool:
    return contains_any_keyword(name, KR_BACKTEST_ETF_NAME_KEYWORDS)


def is_kr_precheck_excluded_security_name(name: str) -> bool:
    return is_kr_backtest_etf_name(name) or contains_any_keyword(
        name, KR_PRECHECK_BAD_SECURITY_KEYWORDS
    )


def is_kr_unwanted_security(item: dict) -> bool:
    """ETF/ETN/SPAC/우선주/리츠 등 백테스트 왜곡 가능성이 높은 종목 제외."""
    name = str(item.get("name", "") if isinstance(item, dict) else "").strip()
    if is_kr_backtest_etf_name(name):
        return True
    if contains_any_keyword(name, KR_BACKTEST_BAD_SECURITY_KEYWORDS):
        return True
    return is_kr_preferred_share_name(name)
