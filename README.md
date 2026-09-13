# MU식 리레이팅 스크리너 v8

"비싸 보여도 추정치·테마·수급이 계속 올라가서 리레이팅될 종목"을 찾는 의사결정 도구.

## 실행 방법

```bash
# 1. WSL 터미널에서:
rm -rf ~/etf-radar-v7-run
mkdir ~/etf-radar-v7-run
unzip /mnt/c/Download/etf-radar-v7.zip -d ~/etf-radar-v7-run

# 2. 가상환경 설치 및 실행:
cd ~/etf-radar-v7-run/etf-radar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

## 탭 구조

| 탭 | 내용 |
|---|---|
| 🎯 리레이팅 후보 | 메인. Re-rating Score 순 통합 랭킹 + 트랙 필터 |
| 🤖 GPT 분석 | 3모드: MU식/저평가/균형형 (두 순위 분리 강제) |
| ⚙️ 고급/진단 | 테마입력, 섹터ETF, 백테스트(감사표+산점도), AI인사이트, 재무 |

## 백테스트

- **워크포워드**: look-ahead 차단, T+1 진입가, ETF 제외
- **Train/Test 분리 최적화**: 70% 학습 / 30% 검증 (과최적화 방지)
- **스냅샷**: 과거 날짜 기준 실제 랭킹 로직으로 복원, SPY/QQQ 초과수익 계산
- 캐시: 일반 초기화 / 완전 초기화 분리

## 주의

본 스크리너는 정량 후보 압축 도구이며 투자 추천이 아닙니다.
최종 매수 판단은 직접 확인 후 본인 책임으로 이루어져야 합니다.
