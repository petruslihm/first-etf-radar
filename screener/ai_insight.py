"""
screener/ai_insight.py  v3

완전 재설계:
- GPT를 "시장 맥락 해석기"로 사용 (종목 추천기 아님)
- JSON 구조화 파싱 (파싱 깨짐 방지)
- Theme Score 계산 (테마 강도 객관화)
- role별 처리 차별화 (leader/direct/related)
- Context Layer (팩터 가중치 동적 조정)
- ON 같은 단어형 티커 파싱 버그 수정
"""

import json
import math
import re
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── 실제 상장된 단어형 티커 화이트리스트 ─────────────────────────
WORD_TICKERS = {
    "A","AA","AB","AC","AI","AM","AN","AY",
    "BE","BY",
    "CF","CG","CI","CO","CS",
    "DB","DD","DG","DH","DK","DL","DM","DO","DQ","DS","DT","DV",
    "EA","EB","ED","EE","EG","EH","EI","EL","EM","EN","EQ","ER","ES","ET","EU","EV",
    "FI","FL","FM","FN","FO","FR","FS","FT","FU","FV",
    "GB","GD","GE","GH","GI","GK","GL","GM","GN","GO","GP","GQ","GR","GS","GT","GU","GV","GW","GX",
    "HI","HK","HL","HM","HN","HO","HP","HQ","HR","HS","HT","HU","HV","HW",
    "IB","IC","ID","IE","IF","IG","IH","II","IJ","IK","IL","IM","IN","IO","IP","IQ","IR","IS","IT","IU","IV",
    "JD","JE","JF","JG","JH","JI","JJ","JK","JL","JM","JN","JO","JP","JQ","JR","JS","JT",
    "KD","KE","KF","KG","KI","KN","KO","KR","KS","KT",
    "LA","LB","LC","LD","LE","LF","LG","LH","LI","LJ","LK","LL","LM","LN","LO","LP","LQ","LR","LS","LT","LU","LV","LW","LX",
    "MA","MB","MC","MD","ME","MF","MG","MH","MI","MJ","MK","ML","MM","MN","MO","MP","MQ","MR","MS","MT","MU","MV","MW","MX","MY",
    "NA","NB","NC","ND","NE","NF","NG","NH","NI","NJ","NK","NL","NM","NN","NO","NP","NQ","NR","NS","NT","NU","NV","NW","NX","NY",
    "OB","OC","OD","OE","OF","OG","OH","OI","OJ","OK","OL","OM","ON","OO","OP","OQ","OR","OS","OT","OU","OV","OW","OX",
    "PA","PB","PC","PD","PE","PF","PG","PH","PI","PJ","PK","PL","PM","PN","PO","PP","PQ","PR","PS","PT","PU","PV","PW","PX","PY",
    "QD","QE","QS",
    "RA","RB","RC","RD","RE","RF","RG","RH","RI","RJ","RK","RL","RM","RN","RO","RP","RQ","RR","RS","RT","RU","RV","RW","RX","RY",
    "SA","SB","SC","SD","SE","SF","SG","SH","SI","SJ","SK","SL","SM","SN","SO","SP","SQ","SR","SS","ST","SU","SV","SW","SX","SY",
    "TA","TB","TC","TD","TE","TF","TG","TH","TI","TJ","TK","TL","TM","TN","TO","TP","TQ","TR","TS","TT","TU","TV","TW","TX","TY",
    "UA","UB","UC","UD","UE","UF","UG","UH","UI","UJ","UK","UL","UM","UN","UO","UP","UQ","UR","US","UT","UV","UW","UX","UY",
    "VA","VB","VC","VD","VE","VF","VG","VH","VI","VJ","VK","VL","VM","VN","VO","VP","VQ","VR","VS","VT","VU","VV","VW","VX","VY",
    "WA","WB","WC","WD","WE","WF","WG","WH","WI","WJ","WK","WL","WM","WN","WO","WP","WQ","WR","WS","WT","WU","WV","WW","WX","WY",
    "XD","XE","XF","XL","XN","XO","XP","XR","XS","XT","XY",
    "YD","YE","YF","YY",
    "ZD","ZE","ZG","ZI","ZM","ZN","ZS","ZT","ZU","ZW",
}

# 명백히 제외할 일반 단어 (위 화이트리스트에 없는 짧은 단어들)
EXCLUDE_WORDS = {
    "THE","AND","FOR","WITH","FROM","THAT","THIS","HAVE","WILL","BEEN",
    "THEY","THEIR","WHAT","WHEN","WHICH","ABOUT","THERE","WOULD","COULD",
}


# ════════════════════════════════════════════════════════════════════
# JSON 기반 테마 프롬프트 생성
# ════════════════════════════════════════════════════════════════════

def generate_theme_scan_prompt(market: str = "US") -> str:
    mkt  = "미국" if market == "US" else "한국"
    tfmt = "티커(예: NVDA)" if market == "US" else "종목코드 6자리(예: 005930)"

    return f"""너는 현재 시장 흐름을 잘 아는 투자 전문가야.

[요청]
지금 {mkt} 시장에서 향후 1~3개월 강하게 움직일 세부 테마와 핵심 종목을 분석해줘.

중요:
- 넓은 테마(AI 반도체)보다 세부 테마(HBM/DRAM, AI 스토리지, 전력 변압기)로 쪼개줘
- 각 테마의 수명주기(early/mid/late)를 반드시 평가해줘
- 각 종목을 leader(직접 최대수혜), direct(직접수혜), related(간접수혜)로 구분해줘
- 지금 현재 시장 맥락에서 어떤 팩터가 더 중요한지도 알려줘

[출력 형식]
반드시 아래 JSON만 출력해. 다른 텍스트 없이 JSON만.

{{
  "market_context": "{mkt} 현재 시장 상황 한 줄 요약",
  "key_factors": {{
    "catalyst": 1.0,
    "momentum": 1.0,
    "volume": 1.0,
    "supply": 1.0,
    "buyable": 1.0
  }},
  "themes": [
    {{
      "name": "세부 테마명",
      "stage": "early",
      "conviction": "high",
      "reason": "이유 한 줄",
      "tickers": [
        {{"ticker": "{tfmt}", "role": "leader", "weight": 1.0}},
        {{"ticker": "티커2", "role": "direct", "weight": 0.8}},
        {{"ticker": "티커3", "role": "related", "weight": 0.5}}
      ]
    }}
  ],
  "avoid_sectors": ["피해야 할 섹터"],
  "avoid_tickers": ["피해야 할 티커"]
}}

key_factors 설명:
- 1.0 = 기본, 1.2 = 이 팩터가 지금 더 중요, 0.8 = 덜 중요
- catalyst: 실적/뉴스 촉매 중요도
- momentum: 차트 모멘텀 중요도
- volume: 거래대금 강도 중요도
- supply: 외국인/기관 수급 중요도 (한국 전용)
- buyable: 진입 자리 중요도

stage: early(초기진입), mid(중반), late(후반주의)
conviction: high(확신), medium(보통), low(불확실)
role: leader(대장주), direct(직접수혜), related(간접수혜)
weight: leader=1.0, direct=0.7~0.9, related=0.3~0.6"""


# ════════════════════════════════════════════════════════════════════
# JSON 파싱 (안정적)
# ════════════════════════════════════════════════════════════════════

def parse_theme_response(response_text: str, market: str = "US") -> list[dict]:
    """
    GPT/Claude JSON 답변 파싱.
    JSON 파싱 실패 시 기존 텍스트 파싱으로 fallback.
    """
    if not response_text or not response_text.strip():
        return []

    # JSON 추출 시도
    parsed = _try_json_parse(response_text)
    if parsed:
        return _normalize_parsed(parsed, market)

    # Fallback: 기존 텍스트 파싱
    logger.warning("JSON 파싱 실패, 텍스트 파싱 시도")
    return _parse_text_fallback(response_text, market)


def _try_json_parse(text: str) -> Optional[dict]:
    """JSON 블록 추출 후 파싱."""
    # ```json ... ``` 블록 제거
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # 전체가 JSON인지 시도
    for attempt in [text, _extract_json_block(text)]:
        if not attempt:
            continue
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            # 주석 제거 후 재시도
            cleaned = re.sub(r'//.*?\n', '\n', attempt)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            try:
                return json.loads(cleaned)
            except Exception:
                pass
    return None


def _extract_json_block(text: str) -> Optional[str]:
    """텍스트에서 JSON 블록({...}) 추출."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def _normalize_parsed(data: dict, market: str) -> list[dict]:
    """파싱된 JSON을 표준 형식으로 변환."""
    themes = []
    raw_themes = data.get("themes", [])

    for th in raw_themes:
        tickers = []
        for t in th.get("tickers", []):
            code = str(t.get("ticker","")).strip().upper()
            code = re.sub(r"[^A-Z0-9\-\.]", "", code) if market=="US" else re.sub(r"[^0-9]","",code)[:6]
            if not code:
                continue
            tickers.append({
                "ticker": code,
                "role":   t.get("role", "related"),
                "weight": float(t.get("weight", 0.5)),
            })

        if not tickers:
            continue

        themes.append({
            "name":       str(th.get("name","")),
            "stage":      str(th.get("stage","mid")),
            "conviction": str(th.get("conviction","medium")),
            "reason":     str(th.get("reason","")),
            "tickers":    tickers,
        })

    return themes


def _parse_text_fallback(text: str, market: str) -> list[dict]:
    """기존 텍스트 파싱 (## 테마N: ... 형식)."""
    themes = []
    current = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.search(r"(?:#{1,3}\s*)?테마\s*\d*\s*[:：]\s*(.+)", line)
        if not m:
            m = re.search(r"^\d+[\.\)]\s*(.+)", line)

        if m:
            if current and current["tickers"]:
                themes.append(current)
            current = {"name":m.group(1).strip().strip("*"), "stage":"mid",
                       "conviction":"medium", "reason":"", "tickers":[]}
            continue

        if current is None:
            continue

        rm = re.search(r"(?:이유|reason|배경)\s*[:：]\s*(.+)", line, re.I)
        if rm:
            current["reason"] = rm.group(1).strip()
            continue

        tm = re.search(r"(?:종목|tickers?|stocks?)\s*[:：]\s*(.+)", line, re.I)
        if tm:
            tickers = _extract_tickers(tm.group(1), market)
            current["tickers"].extend(tickers)
            continue

    if current and current["tickers"]:
        themes.append(current)
    return themes


def _extract_tickers(text: str, market: str) -> list[dict]:
    """텍스트에서 티커 추출 (ON 등 단어형 티커 보존)."""
    result = []
    seen   = set()

    if market == "KR":
        for code in re.findall(r"\b(\d{6})\b", text):
            if code not in seen:
                seen.add(code)
                result.append({"ticker":code,"role":"related","weight":0.5})
    else:
        # NVIDIA(NVDA) 형식에서 NVDA 추출
        for m in re.finditer(r"\w+\(([A-Z]{1,6})\)", text):
            t = m.group(1)
            if t not in seen:
                seen.add(t); result.append({"ticker":t,"role":"related","weight":0.5})

        # 일반 단어 분리
        for word in re.split(r"[\s,/·|·•\-]+", text):
            w = re.sub(r"[^A-Za-z0-9\-\.]","",word).upper()
            if not w or len(w) > 6 or len(w) < 1:
                continue
            if w in seen or w in EXCLUDE_WORDS:
                continue
            # 숫자로만 이루어진 것 제외 (한국 코드가 아닌 경우)
            if w.isdigit():
                continue
            # 화이트리스트 우선, 또는 2자 이상 대문자
            if w in WORD_TICKERS or re.match(r"^[A-Z]{2,6}$", w):
                seen.add(w)
                result.append({"ticker":w,"role":"related","weight":0.5})

    return result


def get_all_theme_tickers(themes: list[dict]) -> list[str]:
    seen = set(); result = []
    for th in themes:
        for t in th.get("tickers",[]):
            code = t if isinstance(t,str) else t.get("ticker","")
            if code and code not in seen:
                seen.add(code); result.append(code)
    return result


def get_context_factors(response_text: str) -> dict:
    """JSON 답변에서 key_factors 추출."""
    parsed = _try_json_parse(response_text)
    if parsed:
        return parsed.get("key_factors", {})
    return {}


def get_avoid_list(response_text: str) -> tuple[list, list]:
    """피해야 할 섹터/티커 추출."""
    parsed = _try_json_parse(response_text)
    if parsed:
        return parsed.get("avoid_sectors",[]), parsed.get("avoid_tickers",[])
    return [], []


# ════════════════════════════════════════════════════════════════════
# Theme Score 계산 (앱이 자동으로)
# ════════════════════════════════════════════════════════════════════

def calc_theme_score(theme: dict, full_df: pd.DataFrame) -> dict:
    """
    테마 강도를 가격/거래량 데이터로 객관적으로 계산.

    theme: {"name":..., "stage":..., "tickers":[{"ticker","role","weight"}]}
    full_df: 전체 유니버스 팩터 DataFrame (ticker × 점수)

    Returns: {
      "theme_score": 0~100,
      "breadth":     상승 종목 비율,
      "avg_ret5":    평균 5일 수익률,
      "avg_ret20":   평균 20일 수익률,
      "vol_surge":   평균 거래량 배수,
      "prox52_avg":  평균 52주 고점 근접도,
      "leaders":     대장주 리스트,
      "stage_signal":자동 판단 stage,
    }
    """
    tickers_info = theme.get("tickers", [])
    if not tickers_info:
        return _empty_theme_score()

    # role별 분류
    all_tickers = [t["ticker"] if isinstance(t,dict) else t for t in tickers_info]
    in_df = [t for t in all_tickers if t in full_df.index]

    if not in_df:
        return _empty_theme_score()

    sub = full_df.loc[in_df]

    # ── 핵심 지표 계산 ─────────────────────────────────────────────
    ret5_vals  = sub["ret_5d"].dropna()  if "ret_5d"  in sub.columns else pd.Series()
    ret20_vals = sub["ret_20d"].dropna() if "ret_20d" in sub.columns else pd.Series()
    vol_vals   = sub["vol_surge"].dropna() if "vol_surge" in sub.columns else pd.Series()
    prox_vals  = sub["week52_prox"].dropna() if "week52_prox" in sub.columns else pd.Series()
    mom_vals   = sub["momentum_score"].dropna() if "momentum_score" in sub.columns else pd.Series()

    avg_ret5  = float(ret5_vals.mean())  if len(ret5_vals)  > 0 else 0
    avg_ret20 = float(ret20_vals.mean()) if len(ret20_vals) > 0 else 0
    avg_vol   = float(vol_vals.mean())   if len(vol_vals)   > 0 else 1.0
    avg_prox  = float(prox_vals.mean())  if len(prox_vals)  > 0 else 75
    avg_mom   = float(mom_vals.mean())   if len(mom_vals)   > 0 else 50

    # Breadth: 20일 수익률 양수 비율
    breadth = float((ret20_vals > 0).mean() * 100) if len(ret20_vals) > 0 else 50

    # 52주 신고가 비율
    high52_ratio = float((prox_vals >= 90).mean() * 100) if len(prox_vals) > 0 else 0

    # 대장주 집중도: top1 수익률 / 평균 수익률
    concentration = 0.0
    if len(ret20_vals) > 1 and avg_ret20 != 0:
        top1 = float(ret20_vals.max())
        concentration = min(1.0, top1 / (abs(avg_ret20) + 1e-8) / len(ret20_vals))

    # ── Theme Score 계산 ─────────────────────────────────────────
    score = 50.0

    # 평균 5일 수익률 (0~20점)
    score += max(-15, min(20, avg_ret5 * 1.5))

    # 평균 20일 수익률 (0~20점)
    score += max(-15, min(20, avg_ret20 * 0.8))

    # Breadth (0~20점)
    score += (breadth - 50) * 0.4

    # 거래량 배수 (0~15점)
    if math.isfinite(avg_vol):
        score += min(15, (avg_vol - 1) * 10)

    # 52주 신고가 근접도 (0~10점)
    score += (avg_prox - 75) * 0.4

    # 대장주 집중도 (0~10점) — 집중 = 한 종목에만 몰림 (약한 신호)
    score -= concentration * 5

    # Stage 보정
    stage = theme.get("stage", "mid")
    if stage == "late":
        score -= 10
    elif stage == "early":
        score += 5

    theme_score = round(max(0, min(100, score)), 1)

    # ── 자동 Stage 판단 ─────────────────────────────────────────
    if avg_ret20 < 5 and breadth < 60:
        stage_signal = "early"
    elif avg_ret20 > 20 and breadth > 80 and avg_prox > 90:
        stage_signal = "late"
    else:
        stage_signal = "mid"

    # ── 대장주 찾기 ─────────────────────────────────────────────
    leaders = []
    if len(in_df) > 0 and "ret_20d" in sub.columns:
        ranked = sub["ret_20d"].dropna().nlargest(3)
        for t in ranked.index:
            role_info = next((x for x in tickers_info if isinstance(x,dict) and x.get("ticker")==t), {})
            leaders.append({
                "ticker": t,
                "role":   role_info.get("role","related"),
                "ret20":  round(float(ranked[t]),2),
                "name":   str(full_df.loc[t,"name"]) if "name" in full_df.columns and t in full_df.index else t,
            })

    return {
        "theme_score":   theme_score,
        "breadth":       round(breadth, 1),
        "avg_ret5":      round(avg_ret5, 2),
        "avg_ret20":     round(avg_ret20, 2),
        "vol_surge":     round(avg_vol, 2),
        "prox52_avg":    round(avg_prox, 1),
        "high52_ratio":  round(high52_ratio, 1),
        "avg_mom":       round(avg_mom, 1),
        "leaders":       leaders,
        "stage_signal":  stage_signal,
        "n_in_universe": len(in_df),
        "n_total":       len(all_tickers),
    }


def _empty_theme_score() -> dict:
    return {
        "theme_score":0,"breadth":50,"avg_ret5":0,"avg_ret20":0,
        "vol_surge":1,"prox52_avg":75,"high52_ratio":0,"avg_mom":50,
        "leaders":[],"stage_signal":"mid","n_in_universe":0,"n_total":0,
    }


# ════════════════════════════════════════════════════════════════════
# AI 인사이트 탭용 기존 함수들 (유지)
# ════════════════════════════════════════════════════════════════════

GRADE_COLOR = {
    "A+":"#4ade80","A":"#34d399","B":"#60a5fa",
    "C":"#fbbf24","D":"#f87171","?":"#64748b",
}

PROMPT_LABELS = {
    "성장주_폭발주": "🚀 성장주/폭발주 찾기",
    "가치투자":      "💎 가치투자 저평가 분석",
    "테마_로테이션": "🔄 테마 로테이션 파악",
}

TYPE_ICON = {
    "성장주_폭발주":"🚀","가치투자":"💎","테마_로테이션":"🔄",
}


def grade_badge_html(grade: str, label: str = "") -> str:
    c   = GRADE_COLOR.get(grade, "#64748b")
    lbl = f"{label} " if label else ""
    return (
        f'<span style="background:{c}22;color:{c};border:1px solid {c}55;'
        f'padding:2px 8px;border-radius:4px;font-size:12px;font-weight:900">'
        f'{lbl}{grade}</span>'
    )


def generate_prompts(top_df: pd.DataFrame, market: str = "US") -> dict[str, str]:
    if top_df is None or top_df.empty:
        return {}

    mkt_name = "미국" if market == "US" else "한국"
    rows = []
    for ticker, row in top_df.iterrows():
        name  = row.get("name",row.get("company",ticker))
        label = f"{ticker}({str(name)[:15]})" if market=="US" else f"{name}({ticker})"
        ret20 = row.get("ret_20d",0); ret60=row.get("ret_60d",0)
        mom   = row.get("momentum_score",0); buy=row.get("buyable_score",0)
        risk  = row.get("top_risk_score",0); vol=row.get("vol_surge",1)
        cls   = row.get("track","")
        prox52= row.get("week52_prox",0)
        rows.append(
            f"- {label} | {row.get('sector','')} | "
            f"20일:{ret20:+.1f}% 60일:{ret60:+.1f}% | "
            f"52주고점:{prox52:.0f}% | "
            f"모멘텀:{mom:.0f} 매수:{buy:.0f} 과열:{risk:.0f} | {cls}"
        )
    stock_list = "\n".join(rows)

    p1 = f"""너는 전문 투자 분석가야.
아래는 {mkt_name} 모멘텀 스크리너 Top 종목이야.

{stock_list}

1. "엔비디아처럼 1~2년 계속 갈 수 있는 종목"을 골라줘
2. 각 종목을 A+/A/B/C/D로 분류해줘
3. 지금 가장 강한 테마 1~2개와 핵심 수혜주를 찾아줘
4. {mkt_name} 시장 주도/소외 섹터를 한 줄로 정리해줘

출력 형식 (파싱용, 반드시 지켜줘):
[종목명/티커] 성장등급: A+ | 코멘트: 이유 한 줄"""

    p2 = f"""너는 가치투자 전문가야.
아래는 {mkt_name} 모멘텀 스크리너 Top 종목이야.

{stock_list}

1. PEG/PBR/성장률 기준 저평가 종목을 찾아줘
2. "워런 버핏이라면 살 종목" 1~2개
3. 실적 하향/가이던스 위험 종목을 경고해줘
4. 각 종목 A+/A/B/C/D 등급

출력 형식:
[종목명/티커] 가치등급: A+ | 코멘트: 이유 한 줄"""

    p3 = f"""너는 섹터 로테이션 전문가야.
아래는 {mkt_name} 모멘텀 스크리너 Top 종목이야.

{stock_list}

1. 지금 돈이 몰리는 테마 TOP3 (초기/중반/후반 판단 포함)
2. 다음에 올 테마 예측
3. 각 종목 테마 로테이션 관점 A+/A/B/C/D

출력 형식:
[종목명/티커] 테마등급: A+ | 코멘트: 이유 한 줄"""

    return {"성장주_폭발주":p1, "가치투자":p2, "테마_로테이션":p3}


def parse_ai_response(
    response_text: str,
    top_df: pd.DataFrame,
    response_type: str = "성장주_폭발주",
    market: str = "US",
) -> dict[str, dict]:
    results: dict[str,dict] = {}
    if not response_text or not top_df is not None:
        return results

    ticker_map: dict[str,str] = {}
    for ticker, row in top_df.iterrows():
        ticker_map[str(ticker).upper()] = ticker
        name = row.get("name",row.get("company",""))
        if name:
            ticker_map[str(name).upper()] = ticker
            ticker_map[str(name)[:6].upper()] = ticker

    grade_keys = ["성장등급","가치등급","테마등급","등급","grade"]

    for line in response_text.split("\n"):
        line = line.strip()
        if not line: continue
        bm = re.search(r"\[([^\]]+)\]", line)
        if not bm: continue
        bracket = bm.group(1).strip()

        grade = None
        for gk in grade_keys:
            gm = re.search(rf"{gk}\s*[:：]\s*([ABCDabcd][+\-]?)", line, re.I)
            if gm: grade=gm.group(1).upper().replace("-",""); break

        comment = ""
        for ck in ["코멘트","comment"]:
            cm = re.search(rf"{ck}\s*[:：]\s*(.+?)(?:\||$)", line, re.I)
            if cm: comment=cm.group(1).strip(); break

        if not grade and not comment: continue

        found = None
        cm6 = re.search(r"\((\d{6})\)", bracket)
        if cm6: found=ticker_map.get(cm6.group(1))
        if not found:
            for word in re.split(r"[\s()/,\.\-]", bracket):
                w=word.strip().upper()
                if w and w in ticker_map: found=ticker_map[w]; break
        if not found:
            bu=bracket.upper()
            best=0
            for key,t in ticker_map.items():
                if key and len(key)>=2 and key in bu and len(key)>best:
                    found=t; best=len(key)

        if found:
            results[found]={"grade":grade or "?","comment":comment,"type":response_type}

    return results
