"""
GPT 후속 프롬프트 모음.

JSON 통합 프롬프트 이후, 사용자가 GPT에게 추가로 던지는 단계별 프롬프트들.
매번 메모장에서 복붙하지 않도록 앱 안에 넣어두고 복사 버튼을 제공한다.

{PORTFOLIO} 자리는 app에서 보유 종목/비중으로 자동 치환된다.
"""

# 각 프롬프트: (제목, 본문)
FOLLOWUP_PROMPTS = [
    (
        "1️⃣ 표 작성 (경쟁력·마진 점수화)",
        "이제 JSON은 됐고, 내가 보기 좋게 표를 만들 거야. "
        "각 종목 옆에다가 글로벌 경쟁력, 경쟁사 중에 그 회사가 몇 위 정도인지, "
        "마진율 등 필요한 정보들을 점수화해서 옆에 같이 적어줘. 시간 오래 걸려도 좋아."
    ),
    (
        "2️⃣ 정밀 재채점 (100점, 회피 제외)",
        "오케이. 근데 너무 점수가 러프하니까 회피는 제외하고, "
        "100점 만점으로 다시 정확히 매겨줘. 시간 오래 걸려도 좋아."
    ),
    (
        "3️⃣ 현금 순위 + 장전량 (최근 이슈 반영)",
        "그리고 이제 15개 사이에 <현금>을 순위에 넣을 건데, "
        "주식시장에 영향을 줄 최근 이슈(예를 들면 스페이스X 상장으로 인한 수급 쏠림, "
        "금리 변화, 연준 교체, 최근에 한 혹은 있을 CPI와 같은 지수 발표 등 "
        "최근의 이벤트들)을 먼저 찾아서 네가 정리해줘. "
        "그리고 그것들을 반영하여 현금의 순위를 정해줘. "
        "그리고 현금이 얼마 정도 장전되어 있으면 좋을지 정해줘. "
        "참고로 나는 원래 모든 투자자금을 주식 100%로 넣는 스타일이야. "
        "지금처럼 정말 큰 이벤트(스페이스X 상장이나 금리 이슈)때는 현금을 장전해놓고."
    ),
    (
        "4️⃣ 실시간 보유/후보 점검 (현재가·뉴스·수급)",
        "현재 나의 투자용 자금이\n\n"
        "<내 포트폴리오>\n"
        "{PORTFOLIO}\n\n"
        "그리고 나머지는 현금이야. 참고로 나는 섹터 겹침 신경 안 써. "
        "지금 장이 진행 중이니까, 내 보유 종목들과 매수 후보와 조건부 후보들의 "
        "각 종목의 현재가와 최신 뉴스, 수급 등 필요한 정보들을 하나씩 찾아봐서 "
        "지금 당장 현재 시점에서 내가 가지고 있는 종목에 대해서 "
        "<추가매수/보유/매도>를 정해줘. "
        "매수 후보인 종목들은 <매수/보류>를 정해줘."
    ),
    (
        "5️⃣ 최종 실행안 + 최종 포트폴리오",
        "이제 아까 네가 찾았던 이벤트들을 고려해서 현금이 얼마나 있어야 할지 계산하고, "
        "내가 현재 있는 주식들과 최종 매수 후보들에 대해 최종적으로 어떻게 할지 "
        "지금 당장 실행할 방안을 정해주고, 그 실행에 따른 최종 포트폴리오를 알려줘."
    ),
]


def build_portfolio_block(holdings: dict, alloc: dict) -> str:
    """보유 종목 dict + 배분 정보를 프롬프트용 텍스트로 변환.
    예:
      - 삼성전자(005930): 비중 50%, 평단 70000원
      - 엔비디아(NVDA): 비중 25%
      (현금 약 25%)
    """
    if not holdings:
        return "(보유 종목 없음 — 포트폴리오 탭에서 입력하세요)"
    lines = []
    # alloc rows로 비중 매핑
    wmap = {r["ticker"]: r for r in alloc.get("rows", [])}
    for tk, h in holdings.items():
        name = h.get("name") or tk
        parts = [f"{name}({tk})"]
        wr = wmap.get(tk, {})
        if wr.get("weight", 0) > 0:
            parts.append(f"비중 {wr['weight_pct']:.0f}%")
        ap = h.get("avg_price")
        if ap:
            parts.append(f"평단 {ap:,.0f}")
        lines.append("- " + ", ".join(parts))
    txt = "\n".join(lines)
    if alloc.get("invested", 0) > 0:
        txt += f"\n(현금 약 {alloc.get('cash_pct', 0):.0f}%)"
    return txt


def render_prompt(body: str, holdings: dict, alloc: dict) -> str:
    """{PORTFOLIO} placeholder를 실제 보유 종목으로 치환."""
    if "{PORTFOLIO}" in body:
        return body.replace("{PORTFOLIO}", build_portfolio_block(holdings, alloc))
    return body
