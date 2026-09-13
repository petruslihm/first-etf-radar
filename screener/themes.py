"""
screener/themes.py
미국/한국 뜨거운 테마 감지
- 테마별 구성종목 맵
- 테마별 평균 20일/60일 수익률 계산
- 상위 뜨거운 테마 추출
"""

import math
import numpy as np

# ── 미국 테마 맵 ────────────────────────────────────────────────────
US_THEME_MAP = {
    "AI 인프라":          ["NVDA","AMD","AVGO","MRVL","ARM","SMCI","DELL","HPE"],
    "AI 소프트웨어":       ["MSFT","GOOGL","META","AMZN","CRM","NOW","ORCL","PLTR","AI","SNOW"],
    "반도체 장비":         ["AMAT","LRCX","KLAC","ASML","TER","ENTG","ONTO"],
    "사이버보안":          ["CRWD","PANW","ZS","FTNT","S","CYBR","OKTA"],
    "클라우드":            ["AMZN","MSFT","GOOGL","CRM","NOW","SNOW","DDOG","MDB"],
    "전력 인프라":         ["VST","NRG","CEG","ETR","XEL","AES","GEV","PWR","WATT"],
    "원전":               ["CEG","CCJ","NNE","SMR","OKLO","BWX"],
    "방산":               ["LMT","RTX","NOC","GD","LHX","HII","AXON"],
    "바이오/제약":         ["LLY","NVO","ABBV","MRK","BMY","AMGN","GILD","REGN","MRNA"],
    "금융":               ["JPM","BAC","GS","MS","BLK","V","MA","AXP"],
    "로보틱스/자동화":     ["ISRG","ABB","ROK","ETN","EMR","FANUY"],
    "구리/원자재":         ["FCX","SCCO","TECK","AA","NEM","GOLD"],
    "에너지":             ["XOM","CVX","COP","EOG","SLB","HAL","MPC"],
}

# ── 한국 테마 맵 ────────────────────────────────────────────────────
KR_THEME_MAP = {
    "HBM/메모리":         ["000660","005930","361610","42700"],
    "AI 반도체":          ["000660","005930","058470","042700","086390"],
    "전력기기/변압기":    ["012690","267260","015760","103590","298040"],
    "원전":              ["071320","090350","017800","298040","259960"],
    "방산":              ["047050","012450","064350","272210","003490"],
    "조선":              ["009540","010140","000720","042660"],
    "2차전지":           ["373220","006400","051910","096770","247540"],
    "화장품/뷰티":       ["090430","051600","003600","161890","002780"],
    "바이오":            ["207940","068270","326030","196170","145020"],
    "로봇":              ["277810","090460","215480","336570","014680"],
    "유리기판":          ["011600","353200","049520"],
    "소프트웨어/AI":     ["035420","035720","263750","293490"],
}


def calc_theme_strength(
    factor_df,       # pd.DataFrame (index=ticker)
    theme_map: dict,
    market: str = "US",
) -> list[dict]:
    """
    테마별 평균 수익률·모멘텀을 계산해 뜨거운 테마 순위 반환.
    Returns: [{"theme": str, "avg_ret20": float, "avg_mom": float,
               "top_stocks": [(ticker, name, ret20)], "hot": bool}, ...]
    """
    ret_col  = "ret_20d"
    mom_col  = "momentum_score"
    name_col = "name" if market == "KR" else "company"

    results = []
    for theme, members in theme_map.items():
        # 유니버스에 있는 종목만
        in_universe = [t for t in members if t in factor_df.index]
        if len(in_universe) < 2:
            continue

        sub = factor_df.loc[in_universe]
        avg_ret20 = float(sub[ret_col].dropna().mean()) if ret_col in sub.columns else 0
        avg_mom   = float(sub[mom_col].dropna().mean()) if mom_col in sub.columns else 0

        # 상위 3종목
        if ret_col in sub.columns:
            top3 = sub[ret_col].dropna().nlargest(3)
            top_stocks = [
                (
                    t,
                    str(sub.loc[t, name_col]) if name_col in sub.columns else t,
                    float(sub.loc[t, ret_col]),
                )
                for t in top3.index
            ]
        else:
            top_stocks = []

        results.append({
            "theme":      theme,
            "avg_ret20":  round(avg_ret20, 2),
            "avg_mom":    round(avg_mom, 1),
            "top_stocks": top_stocks,
            "count":      len(in_universe),
            "hot":        avg_ret20 >= 5.0 and avg_mom >= 55,
        })

    # 평균 수익률 기준 정렬
    results.sort(key=lambda x: x["avg_ret20"], reverse=True)
    return results
