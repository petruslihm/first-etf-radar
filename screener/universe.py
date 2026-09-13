"""
screener/universe.py
유니버스 정의 / 제외 필터 / 점수 가중치 / 섹터 정규화
"""

US_BENCH = {
    "market":    "SPY", "growth":    "QQQ", "semis":     "SMH",
    "semis2":    "SOXX","tech":      "XLK", "software":  "IGV",
    "industry":  "XLI", "utility":   "XLU", "finance":   "XLF",
    "energy":    "XLE", "health":    "XLV", "comm":      "XLC",
    "cons_disc": "XLY", "cons_stap": "XLP", "materials": "XLB",
    "realestate":"XLRE",
}

SECTOR_ETF = {
    "Technology":             "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Health Care":            "XLV",
    "Financials":             "XLF",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Energy":                 "XLE",
}

# yfinance 비표준 → GICS 표준 정규화
_NORM = {
    "healthcare":             "Health Care",
    "health care":            "Health Care",
    "financial services":     "Financials",
    "financial":              "Financials",
    "consumer cyclical":      "Consumer Discretionary",
    "consumer defensive":     "Consumer Staples",
    "basic materials":        "Materials",
    "real estate":            "Real Estate",
    "communication services": "Communication Services",
    "utilities":              "Utilities",
    "energy":                 "Energy",
    "industrials":            "Industrials",
    "technology":             "Technology",
    "information technology": "Technology",
}

def normalize_sector(raw: str) -> str:
    if not raw:
        return ""
    key = raw.lower().strip()
    result = _NORM.get(key, raw.strip())
    if result in SECTOR_ETF:
        return result
    for std in SECTOR_ETF:
        if std.lower() == result.lower():
            return std
    return result

UNIVERSE_ETFS = ["SMH","SOXX","XLK","IGV","XLI","XLU","XLF","XLE","XLV"]

US_FILTER = {
    "min_price":        5.0,
    "min_market_cap":   2_000_000_000,
    "min_avg_dv":       50_000_000,
    "min_history_days": 60,
}

KR_FILTER = {
    "min_market_cap":    300_000_000_000,
    "min_avg_amount":    5_000_000_000,
    "min_history_days":  60,
    "exclude_preferred": True,
}

US_WEIGHTS = {
    "momentum":   0.33,
    "volume":     0.15,
    "sector_str": 0.10,
    "buyable":    0.20,
    "quality":    0.10,
    "top_risk":  -0.15,
    # regime haircut은 engine에서 별도 적용
}

KR_WEIGHTS = {
    "momentum":   0.28,
    "volume":     0.12,
    "sector_str": 0.08,
    "supply":     0.15,
    "buyable":    0.20,
    "quality":    0.10,
    "top_risk":  -0.15,
}

def classify(mom: float, buy: float, risk: float) -> str:
    if risk >= 80:
        if risk >= 88:              return "Avoid / Too Extended"
        if mom >= 75:               return "Strong Momentum / Extended"
        return "Avoid / Too Extended" if buy <= 5 else "Strong Momentum / Extended"
    if risk >= 60:
        return "Strong Momentum / Extended" if mom >= 55 else "Avoid / Too Extended"
    if mom < 28:                    return "Avoid / Weak Relative Strength"
    if mom >= 50 and buy >= 55 and risk < 50: return "Strong Momentum / Buyable"
    if mom >= 40 and buy >= 65 and risk < 45: return "Breakout Candidate"
    if mom >= 55 and risk >= 45:    return "Strong Momentum / Extended"
    if mom >= 30 and buy >= 60 and risk < 55: return "Pullback Candidate"
    if mom >= 42:                   return "Watch Only"
    return "Avoid / Weak Relative Strength"
