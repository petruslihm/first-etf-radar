"""
data/us_universe.py
강제 포함 유니버스 — S&P500/Nasdaq100 자동 수집 실패 시 fallback
+ 테마별 핵심 종목 (신규 편입/스핀오프/급부상 종목 포함)

FORCED_UNIVERSE_META: 종목별 added_date 기록
  → 스냅샷 백테스트 시 "그 날짜에 이 앱이 이 종목을 알고 있었는가?" 검증
  → 추가일 이전 스냅샷에서는 자동 제외 (사후편입 편향 방지)
"""

# ── 종목별 편입일 기록 (사후편입 편향 방지) ─────────────────────────
# added_date: 이 앱/유니버스에 처음 추가된 날짜
# 이 날짜 이전의 스냅샷 백테스트에서는 해당 종목 제외
FORCED_UNIVERSE_META = {
    # 메모리/스토리지
    "MU":   {"added_date": "2020-01-01", "theme": "memory",   "reason": "D램 대장주"},
    "WDC":  {"added_date": "2020-01-01", "theme": "storage",  "reason": "HDD/NAND"},
    "STX":  {"added_date": "2020-01-01", "theme": "storage",  "reason": "HDD 대장"},
    "SNDK": {"added_date": "2026-04-20", "theme": "memory",   "reason": "WDC 스핀오프, Nasdaq-100 편입"},
    "SIMO": {"added_date": "2023-01-01", "theme": "memory",   "reason": "낸드 컨트롤러"},
    # AI 인프라
    "NVDA": {"added_date": "2020-01-01", "theme": "ai_infra", "reason": "GPU 대장"},
    "AMD":  {"added_date": "2020-01-01", "theme": "ai_infra", "reason": "AI GPU/CPU"},
    "AVGO": {"added_date": "2020-01-01", "theme": "ai_infra", "reason": "AI 네트워크 칩"},
    "MRVL": {"added_date": "2022-01-01", "theme": "ai_infra", "reason": "AI 맞춤 칩"},
    "ARM":  {"added_date": "2023-09-14", "theme": "ai_infra", "reason": "2023 IPO"},
    "PLTR": {"added_date": "2020-09-30", "theme": "ai_infra", "reason": "데이터 AI"},
    "SMCI": {"added_date": "2023-01-01", "theme": "ai_infra", "reason": "AI 서버"},
    "DELL": {"added_date": "2020-01-01", "theme": "ai_infra", "reason": "AI 인프라"},
    "AI":   {"added_date": "2020-12-09", "theme": "ai_infra", "reason": "기업 AI 플랫폼"},
    # 전력 인프라
    "VST":  {"added_date": "2024-01-01", "theme": "power",    "reason": "AI 전력 수요"},
    "CEG":  {"added_date": "2022-02-01", "theme": "power",    "reason": "원자력 재가동"},
    "NRG":  {"added_date": "2023-01-01", "theme": "power",    "reason": "전력 공급"},
    "GEV":  {"added_date": "2024-04-02", "theme": "power",    "reason": "GE 버노바 분사"},
    "PWR":  {"added_date": "2023-01-01", "theme": "power",    "reason": "전력망 공사"},
    "WATT": {"added_date": "2024-01-01", "theme": "power",    "reason": "에너지 효율"},
    "EME":  {"added_date": "2023-01-01", "theme": "power",    "reason": "전기 공사"},
    # 사이버보안
    "CRWD": {"added_date": "2019-06-12", "theme": "cyber",    "reason": "엔드포인트 보안"},
    "PANW": {"added_date": "2020-01-01", "theme": "cyber",    "reason": "네트워크 보안"},
    "ZS":   {"added_date": "2020-01-01", "theme": "cyber",    "reason": "제로트러스트"},
    "NET":  {"added_date": "2019-09-13", "theme": "cyber",    "reason": "클라우드 보안"},
    "CYBR": {"added_date": "2020-01-01", "theme": "cyber",    "reason": "ID 보안"},
    "FTNT": {"added_date": "2020-01-01", "theme": "cyber",    "reason": "방화벽"},
    "S":    {"added_date": "2021-06-30", "theme": "cyber",    "reason": "AI 보안"},
    # 방산
    "LMT":  {"added_date": "2020-01-01", "theme": "defense",  "reason": "방산 대장"},
    "RTX":  {"added_date": "2020-01-01", "theme": "defense",  "reason": "방산/항공"},
    "AXON": {"added_date": "2022-01-01", "theme": "defense",  "reason": "경찰/무인"},
    "HII":  {"added_date": "2020-01-01", "theme": "defense",  "reason": "해군 조선"},
    "CACI": {"added_date": "2023-01-01", "theme": "defense",  "reason": "방산 IT"},
    # 바이오
    "LLY":  {"added_date": "2020-01-01", "theme": "biotech",  "reason": "비만약 대장"},
    "VRTX": {"added_date": "2020-01-01", "theme": "biotech",  "reason": "희귀질환"},
    "REGN": {"added_date": "2020-01-01", "theme": "biotech",  "reason": "항체치료제"},
}

# ── 메모리/스토리지 ──────────────────────────────────────────────
MEMORY_STORAGE = ["MU","SNDK","WDC","STX","SIMO","MRVL","NVDA","AMD","AVGO","INTC"]

# ── AI 인프라 ────────────────────────────────────────────────────
AI_INFRA = ["NVDA","AMD","AVGO","MRVL","ARM","SMCI","DELL","HPE","PLTR","AI"]

# ── 전력 인프라 ──────────────────────────────────────────────────
POWER_INFRA = ["VST","CEG","ETR","NRG","AES","GEV","PWR","WATT","EME","URI"]

# ── 사이버보안 ───────────────────────────────────────────────────
CYBERSEC = ["CRWD","PANW","ZS","FTNT","S","CYBR","OKTA","NET"]

# ── 방산 ─────────────────────────────────────────────────────────
DEFENSE = ["LMT","RTX","NOC","GD","LHX","HII","AXON","CACI"]

# ── 바이오/제약 ──────────────────────────────────────────────────
BIOTECH = ["LLY","NVO","ABBV","MRNA","REGN","AMGN","GILD","VRTX"]

# 전체 강제 포함 유니버스 (현재 실행용)
FORCED_UNIVERSE = list(set(
    MEMORY_STORAGE + AI_INFRA + POWER_INFRA +
    CYBERSEC + DEFENSE + BIOTECH
))


def get_forced_universe_at(snapshot_date: str) -> list[str]:
    """
    특정 날짜 기준으로 유니버스에 포함 가능한 종목만 반환.
    added_date > snapshot_date 인 종목은 제외 (사후편입 편향 방지).
    """
    from datetime import date as _date
    try:
        snap_d = _date.fromisoformat(snapshot_date)
    except Exception:
        return FORCED_UNIVERSE  # 날짜 파싱 실패 시 전체 반환

    result = []
    for ticker in FORCED_UNIVERSE:
        meta = FORCED_UNIVERSE_META.get(ticker)
        if meta is None:
            result.append(ticker)  # 메타 없으면 포함 (보수적)
            continue
        try:
            added = _date.fromisoformat(meta["added_date"])
            if added <= snap_d:
                result.append(ticker)
        except Exception:
            result.append(ticker)
    return result

# S&P500 fallback — 2025년 기준 현재 구성종목 (Wikipedia 수집 실패 시)
# 상장폐지 종목 제거: CERN(인수), ALXN(인수), DISH(인수), WBA(상폐), SPLK(인수),
#   ENDP(파산), FLIR(인수), MXIM(인수), MYL(합병), NUAN(인수), SGEN(인수) 등
SP500_FALLBACK = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","AVGO","JPM",
    "LLY","UNH","XOM","V","COST","MA","PG","JNJ","HD","ABBV",
    "MRK","CVX","WMT","BAC","KO","PEP","ORCL","ACN","CRM","AMD",
    "MU","NFLX","TMO","ABT","CSCO","WFC","TXN","AMGN","GE","DHR",
    "IBM","ISRG","INTU","QCOM","CMCSA","NEE","UPS","RTX","HON","LOW",
    "SPGI","NOW","BKNG","GS","BLK","AMAT","LRCX","KLAC","ADI","PANW",
    "SYK","PLD","DE","ETN","AXP","C","VRTX","REGN","MDT","ZTS",
    "GILD","MMC","TJX","MDLZ","SCHW","PGR","CTAS","CB","ITW","AON",
    "NOC","GD","LMT","RTX","HII","LHX","LDOS","SAIC","CACI","AXON",
    "CRWD","FTNT","PANW","ZS","OKTA","S","CYBR","NET","DDOG","SNOW",
    "PLTR","UBER","LYFT","ABNB","DASH","RBLX","COIN","SQ","PYPL","SHOP",
    "VST","CEG","NRG","ETR","AES","GEV","PWR","EME","MYRG","WATT",
    "IBB","MRNA","REGN","VRTX","GILD","ALNY","EXAS","NTRA","RXRX",
    "SMCI","DELL","HPE","NTAP","PURE","PSTG","STX","WDC","SNDK","MU",
    "MRVL","ARM","INTC","QCOM","TXN","AMAT","LRCX","KLAC","ASML","TER",
    "FCX","NEM","GOLD","AA","CLF","X","NUE","STLD",
    "GS","MS","BAC","WFC","C","JPM","AXP","BLK","SPGI","CME","ICE",
    "DIS","NFLX","CMCSA","WBD","PARA","FOX",
    "AMZN","EBAY","ETSY","W","CHWY","CPNG",
    "CVS","UNH","HUM","ELV","CI","CNC","MOH",
    "CAT","DE","HON","ETN","EMR","ROK","ITW","PH","AME","FTV",
]
