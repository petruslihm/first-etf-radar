# 개선 작업 요약 (v8 → v8.1)

목적: 새 기능 추가가 아니라 **기존 설계의 정확성·견고성 구멍을 메워 완성도를 올리는 것**.
모든 변경은 컴파일 + 단위테스트(19개 통과)로 검증됨. 라이브 Streamlit/네트워크는
이 환경에서 실행 불가하여, 라이브 경로 변경은 보수적으로만 적용함.

---

## 1순위 — 점수 공식 일원화 (가장 중요)

"백테스트가 실전 점수 공식을 검증한다"는 이 프로그램의 핵심 전제가 4곳에서 깨져 있었음.
실전 점수(`calc_validated_score_series`)는 enhanced 컬럼(volume_enhanced 등)과 양수
가중치 정규화를 쓰는데, 아래 4곳은 raw 수식을 손으로 재구현해 어긋나 있었음.

- `screener/backtest.py · optimize_weights._sharpe_top` (US 가중치 최적화)
  → 공유 함수 `calc_validated_score_series` 호출로 교체.
- `screener/backtest.py · optimize_kr_weights._sharpe_kr` (KR 가중치 최적화)
  → 동일하게 교체.
- `screener/backtest.py · run_snapshot_backtest` 의 composite 점수
  → 공유 함수로 교체.
- `screener/backtest.py · run_snapshot_backtest` 의 트랙 분류
  → 손으로 재구현(이미 `classify_track`과 미세하게 drift됨)을 제거하고
    `screener.factors.classify_track`을 직접 호출.

영향: 이제 튜닝한 "최적 가중치"가 실제 배포 공식 기준으로 최적화됨.
부수: 점수/트랙 로직이 바뀌었으므로 `BT_VERSION`을 v10.5로 올려 과거 백테스트/
스냅샷 캐시를 자동 무효화. (가중치 캐시는 다음 최적화 실행 시 갱신됨)

## 2순위 — 견고성 (조용한 실패 제거)

- `screener/collector.py`: 네트워크 재시도 헬퍼 `_get_with_retry` 추가
  (지수 백오프 3회, 4xx는 즉시 중단·429만 재시도). 네이버 호출 5곳을 모두
  이 헬퍼로 교체하고 None 가드 추가 → 사이트 일시 오류 시 종목이 통째로
  조용히 사라지던 문제 완화. 실패는 `logger.warning`으로 남김.
- `app.py`: 수집/필터 실패 종목이 세션에만 저장되고 화면엔 전혀 안 보이던 문제 수정.
  US/KR 스캔 완료 직후 제외된 종목 수와 목록(최대 30개)을 `st.caption`으로 노출.

(남은 ~80여 개 `except: pass`는 대부분 캐시 쓰기 등 비핵심 경로라, 위험 대비
이득이 낮아 일괄 수정하지 않음.)

## 3순위 — 정리

- **죽은 ETF 서브시스템 6개 모듈 삭제 (총 1,226줄)** — 현재 파이프라인에서 참조 0회:
  - `utils/scorer.py`, `utils/theme_scorer.py`, `utils/theme_mapper.py`
  - `utils/fetcher.py`, `utils/market_scanner.py`
  - `data/etf_universe.py`
  삭제 후 전체 재컴파일로 import 깨짐 없음 확인.
- 비정상 디렉토리 `{data/...` (brace expansion 실패 흔적) 제거.
- `calc_us_factors` / `calc_kr_factors` **공통부 추출 완료** (특성화 테스트로 안전 검증).
  두 함수에 그대로 중복돼 있던 ① `bret` 벤치마크 수익률 클로저 ② 8개 코어 점수
  계산 블록 ③ `sect_str` 섹터강도 공식을, 공유 헬퍼 3개로 단일화:
  `_bench_ret()`, `_core_factor_scores()`, `_sector_strength()`.
  시장별로 다른 부분(벤치마크 키, MA 윈도우 50d vs 60d, 수급·신규팩터,
  출력 키, 가중치 로딩)은 각 함수에 그대로 보존 → 정당한 분기는 합치지 않음.

  **안전 검증 방식**: 라이브 실행이 불가하므로 "특성화(characterization) 테스트"로
  처리. `tools/char_factors.py`가 결정적 합성 OHLCV로 리팩터 **전** 두 함수의
  전체 출력(46개 필드 × 3케이스 × US/KR)을 골든 스냅샷에 박제 → 리팩터 **후**
  출력이 100% 동일함을 확인. 이 검증은 `tests/test_characterization.py`로
  영구 회귀 테스트화되어, 앞으로 두 함수를 의도치 않게 바꾸면 자동 실패함.

## 4순위 — 안전망 & 마감

- `tests/` 신규: 단위 + 특성화 테스트 21개 (네트워크 불필요).
  - `test_scoring.py`: risk penalty 임계값, 스칼라↔시리즈 일관성,
    enhanced 컬럼 우선, 가중치 정규화 불변식, **1순위 회귀 가드**.
  - `test_factors.py`: clamp/rs/ret 헬퍼, classify_track 경계값,
    **look-ahead 슬라이싱 불변식**(미래 데이터 차단 검증).
  - `test_characterization.py`: calc_us/kr_factors 출력 골든 비교(리팩터 회귀 가드).
  - 실행: `python3 -m pytest tests/ -v`
- `requirements.txt`: 전부 `>=`였던 것에 **메이저 버전 상한** 추가
  (yfinance/streamlit 등 잦은 업데이트로 인한 예기치 않은 파괴적 변경 차단).

---

## 남은 권장 작업 (이번에 안 한 것)

- 프로젝트/폴더명을 실제 정체성(개별주 리레이팅 스크리너)에 맞게 변경 — 사용자 판단 필요.
- KR 애널리스트 추정치 비대칭 — 무료 데이터 소스가 마땅치 않아 보류(기능 추가 영역).
- 로컬에서 실제 앱(`streamlit run app.py`)을 한 번 돌려 라이브 경로까지 최종 확인 권장.

---

# v8.2 — 한국 백테스트 결과 기반 수정

한국 백테스트 결과(합성점수 Spearman≈-0.02, 모든 분위 KOSPI 미달, quality 역효과)를
근거로 한 수정. 모든 변경은 특성화 테스트 + 단위테스트(22개)로 검증.

## quality 팩터 제거
백테스트에서 quality(추세 품질)가 수익률 상관 -0.041, 급등 적중 ~0으로 **양쪽 모두
역효과** 판정. 라이브·백테스트·옵티마이저 전 경로에서 일관되게 제거:
- `utils/scoring.py` DEFAULT_WEIGHTS["quality"] → 0.0
- `screener/factors.py` calc_us_factors / calc_kr_factors: 캐시 가중치까지 덮어 강제 0
- `screener/backtest.py` load_optimal_weights / _kr / _default_weights: quality 강제 0
- `screener/backtest.py` 옵티마이저 그리드(US/KR): quality 후보 0 고정
- BT_VERSION v10.6 → 백테스트 캐시 자동 무효화
- 특성화 테스트 결과 leader_final/buyable_final/validated_score/final_score만 변경,
  나머지 42개 필드 불변 확인. 골든 재생성.
- 회귀 가드: tests/test_scoring.py::test_quality_factor_removed
- ⚠️ 단일 기간 결과 기반. 적용 후 "⚡ 점수만 재계산"으로 백테스트 재실행해
  분위 단조성·Spearman이 실제 개선됐는지 반드시 확인할 것. 악화 시 되돌릴 것.

## ⚡ "점수만 재계산" 버튼 추가 (한국 재수집 회피)
문제: 기존 일반/완전 초기화가 둘 다 kr_ohlcv 캐시를 지워, 가중치만 바꿔도
한국 전종목을 재스크래핑(종목당 수십 페이지 순차 → 수십 분)했음.
- `screener/engine.py` clear_scores_cache() 신규: factors/sector 캐시만 삭제,
  비싼 kr_ohlcv 수집 캐시는 보존.
- `app.py` 고급/진단 탭: "⚡ 점수만 재계산" 버튼 추가(점수+백테스트만 재계산,
  수집 데이터 유지) → 가중치 실험이 수십 분에서 수 초로 단축.

## 수집 속도 — OHLCV 핫패스 재시도 완화
`_naver_ohlcv` 페이지 루프(종목당 수십 회)의 재시도를 1회로 제한.
페이지 1개 실패는 비치명적(건너뜀)이라, v8.1에서 넣은 백오프 재시도가
핫패스에서 일으키던 속도 저하를 제거. 단발 중요 호출은 재시도 3회 유지.

## 한국이 미국보다 느린 이유 (구조적)
- 미국: yfinance 배치 다운로드(1회 호출로 다수 종목) → 빠름.
- 한국: 네이버를 종목당 ~52페이지 순차 스크래핑(외부 루프만 8스레드 병렬,
  페이지 루프는 순차) → 수백 종목 × 52요청 = 수만 요청. 구조적으로 느림.
- 완화책: 위 "점수만 재계산"으로 재수집 자체를 회피하는 게 핵심.

---

# v8.3 — 영구 캐시 + 데이터 기준일 표시 (zip 교체에도 보존)

문제: 캐시가 zip 압축 해제 폴더 안(`data/cache`)에 있고 파일명에 날짜가 박혀 있어,
코드를 새 zip으로 갈아끼우면 그날 수집 데이터가 사라지고, 자정만 넘어도 무효화됨.
→ 매번 한국 전종목 재수집(수십 분).

## 캐시를 zip 바깥 고정 위치로
- `screener/paths.py` 신규: 캐시 경로/기준일 중앙 관리.
  - 저장 위치 = 환경변수 `ETF_DATA_DIR` (없으면 `~/.etf-radar-cache`).
    → zip을 새로 풀어도 보존됨. 원하는 드라이브로 바꾸려면 ETF_DATA_DIR 설정.
  - collector/engine/backtest 의 흩어진 CACHE_DIR 정의 3곳을 이 한 곳으로 통일.

## 날짜 키 제거 + 데이터 기준일 스탬프
- OHLCV per-stock 캐시(`kr_ohlcv_{code}_{pages}p.pkl`): 날짜 키 제거 → 영구 보존.
- "수집한 날짜"는 별도 스탬프 파일(`data_stamp.txt`)에 기록.
  최초 수집 시 1회 기록, 이후 재사용 시 유지, 완전 초기화 시 리셋.
- 파생 캐시(factors/sector/tickers/dynamic_universe)는 스탬프 날짜로 키잉
  → 데이터가 갱신되면 자동 무효화(오래된 점수가 새 데이터에 붙는 버그 방지).

## 데이터 기준일 표시 (요청 기능)
- 사이드바 상단 배지: "📅 데이터 기준일: YYYY-MM-DD (오늘 / N일 전)".
  N일 전이면 최신화 방법 안내.
- ⚙️고급/진단 캐시 섹션: 기준일 + 캐시 저장 위치 경로 표시.

## 캐시 초기화 동작 (스탬프 반영)
- ⚡ 점수만 재계산: factors/sector만 삭제, OHLCV·기준일 보존 → 재수집 없이 수 초.
- ↺ 일반 / 🔥 완전 초기화: OHLCV 포함 삭제 + 기준일 리셋 → 다음 스캔이 오늘 데이터로 재수집.
- 검증: 통합 테스트로 "점수만=보존 / 완전=리셋" 동작 확인.

## 실험 하니스(tools/experiment.py)도 공용 캐시 사용
- 3단계 캐시(L1 스냅샷 / L2 백테스트 레코드 / L3 가중치 즉시 실험).
- L3은 기존 레코드로 합성점수만 재계산 → 가중치 실험이 네트워크·백테스트 없이 즉시.
  예: python3 tools/experiment.py weights kr --w leader=0.5,volume=0.2,...

## 사용 시 주의
- v8.3 첫 실행은 캐시 위치가 ~/.etf-radar-cache 로 바뀌어 1회 재수집 발생(정상).
- 이후 같은 데이터 기준일 동안은 zip을 새로 풀어도 재수집 없이 빠름.

---

# (v8.3 zip 유지) 보유기간 실험 + GPT 후보수 변경

파일명은 요청에 따라 etf-radar-v8.3.zip 그대로 유지. 점수 로직은 안 바뀌어
캐시(OHLCV) 재사용되므로 빠름.

## 보유기간(lookahead) 스윕 — tools/experiment.py
배경: 리레이팅은 몇 달에 걸친 멀티플 확장 가설인데 백테스트는 20일만 측정.
시간축 불일치가 신호 0의 원인일 수 있어, 여러 보유기간을 비교하는 기능 추가.
- `backtest`/`report`에 `--la` 인자 추가(기본 20). 레코드는 기간별로 캐시
  (`_bt_records_{market}_la{N}.pkl`)되어 한 번 만들면 재사용.
- 신규 `sweep` 명령: 여러 기간을 한 번에 비교(Spearman·상위20%초과·분위 단조).
  예) python3 tools/experiment.py sweep kr --horizons 20,60,120
  (스냅샷 재사용 → 재수집 없음. 기간별 백테스트만 1회씩.)
- 해석: 더 긴 기간에서 Spearman이 +로 뚜렷해지거나 단조면, 전략 시간축이
  20일보다 길다는 뜻 → 운영 기간 재고.

## GPT 리레이팅 창 — 최대 후보수 = 걸러낸 후보수
- 기존: "최대 후보 수" 슬라이더가 6~24 고정(기본 18)이라 후보를 임의로 잘랐음.
- 변경: 슬라이더 상한·기본값을 '걸러낸 후보 수'로 설정 → 기본적으로 통과한
  후보 전체를 GPT로 보냄. 줄이고 싶으면 슬라이더로 조절 가능.
  후보가 6개 이하면 슬라이더 없이 전체 사용.
- 재스캔으로 후보 수가 바뀌어 이전 세션값이 범위를 벗어나면 자동 초기화.

## 참고
- 보유기간이 60/120일에서 더 좋게 나오면, 앱 백테스트도 그 기간을 쓰도록
  바꾸는 후속 작업 가능(현재 앱은 20일 고정).

---

# (v8.3 zip 유지) 수급 백테스트 편입 — 1단계: 측정

진단 결과 네이버가 외국인/기관 일별 순매매를 ~800거래일(약 40개월) 제공 →
백테스트 전 구간 커버 가능. 수급을 백테스트에 look-ahead 없이 편입.

## 날짜별 수급 시계열 수집 (collector.py)
- `_naver_supply_series(code, pages=30)`: frgn 페이지를 여러 장 긁어 날짜 인덱스 +
  [foreign, inst] 일별 순매매 DataFrame 생성. 캐시(`kr_supply_{code}_{pages}p.pkl`).
- `attach_kr_supply_series(data)`: 유니버스 각 종목에 item['supply_series'] 부착(병렬).

## 백테스트 점수 시점(T) 기준 수급 주입 (backtest.py) — look-ahead 차단
- `_supply_flow_at(sup_df, decision_dt, mc, avg_amt)`: 결정일 T까지의 수급만 잘라
  5일/20일 순매매 합 → kr_supply_score로 0~100 점수. 미래 수급 사용 안 함.
- run_kr_walkforward_backtest: item에 supply_series 있으면 각 레코드의
  supply_flow를 T시점 값으로 채움(없으면 기존대로 50).
- 단위 테스트: 미래 수급을 오염시켜도 T시점 점수 불변(test_supply_flow_at_no_lookahead).
- 합성 데이터 end-to-end 확인: supply_flow가 9~100으로 변동, 순매수>순매도 방향 정상.

## 실험 절차 (중요: 측정 먼저, 가중치는 그 다음)
supply_flow는 점수 공식상 기본 가중치 0 → 레코드 컬럼으로만 먼저 채운다.
1) snapshot kr --supply   : 수급 시계열까지 수집(최초 1회 느림, 이후 캐시)
2) backtest kr --la 20     : 수급 포함 레코드 재생성
3) report kr --la 20       : supply_flow 단독 Spearman 확인 (예측력 있나?)
4) 예측력 있으면 weights로 가중치 부여 테스트:
   weights kr --w "leader=0.33,volume=0.18,breakout=0.10,catalyst=0.12,risk_penalty=0.10,supply_flow=0.20"
- 앱 백테스트는 아직 20일/수급 미편입. 수급이 알파를 더한다고 확인되면 그때 앱에 반영.

---

# (v8.3 zip 유지) C: GPT 정성 점검 흐름 (급등 레이더 + 정성 판단)

백테스트 결론(수익 예측 X, 급등 탐지 O)에 맞춰 GPT 분석을 재정비.
점수의 역할을 "수익 예측"에서 "후보 발굴"로 낮추고, 매수 판단은 GPT+사람 정성.

## 새 GPT 모드: "🔍 급등 후보 점검 (단기)" (기본 선택)
- 프롬프트가 솔직히 명시: "이 목록은 오를 종목이 아니라 변동성이 터질 후보다.
  방향은 점수가 못 맞히니 정성 판단해달라."
- 5단계 정성 점검: ①실제 촉매 ②수급(외국인/기관) ③초입 vs 끝물(한국 평균회귀 경계)
  ④빠질 신호/함정 ⑤단타 대응(진입/관망/회피 + 손절선 + 목표 + 보유기간).
- 출력에 손절선·대응표·"오늘 대응할 1~3개 / 만지지 말 것" 포함.
- 정렬: 급등(hit15) 예측 팩터(돌파0.4+과열0.3+주도0.3) 기준.

## 종목 블록에 수급 정보 추가 (KR)
- 프롬프트의 각 종목 줄에 수급점수·외국인5일 방향 추가 → GPT가 한국 핵심
  변수인 수급을 직접 보고 판단.

## 기존 3개 모드(리레이팅/저평가/균형)는 그대로 유지.

---

# (v8.3 zip 유지) 기록 & 검증 저널 (예측 → 자동 채점)

GPT 단기 대응 추천을 날짜별로 저장하고, 며칠 뒤 실제 가격으로 자동 채점.

## screener/journal.py (신규)
- 저장: zip 바깥 ~/.etf-radar-cache/journal/picks_{market}_{날짜}.json (코드 갱신과 무관·영구).
- parse_gpt_json: GPT 텍스트에서 JSON 추출(```json 펜스/앞뒤 설명 섞여도 처리, 한글키 허용).
- capture_entry_prices: 저장 시 기준일 종가를 진입가로 박제(per-stock OHLCV 캐시에서).
- score_entry: look-ahead 없이 채점 — 진입일 '이후' N거래일 종가로 수익률,
  손절선은 구간 저가가 깼는지로 판정. 데이터 부족 시 '대기'.
- 판정: 진입→✅성공/❌손절/❌부진/➖중립, 관망·회피→⚠️놓침/✅회피적중/➖.
- summary: 진입 승률·평균수익.

## app.py — GPT 탭에 기록/검증 UI
- "🧾 통합 JSON 요청 프롬프트": GPT에 마지막으로 보내 고정 JSON을 받게 함(매일 형식 통일).
- "📒 기록 & 검증": JSON 붙여넣기→저장(진입가 자동 박제), 날짜 선택, 평가기간(거래일) 조절,
  채점 표(진입가/평가가/수익%/손절/판정/상태) + 진입 승률 요약, 날짜별 삭제.

## 4가지 요구 대응
1) 빠진 날: 채점은 날짜에 안 묶임 — 나중에 열면 그 사이 실제 가격으로 소급 채점.
2) 업데이트 보존: journal/ 이 zip 바깥이라 코드 갱신해도 전부 유지.
3) 종목수 유동: picks 배열 — 몇 개든 저장/채점.
4) 쉬운 입력/비교: JSON 붙여넣기 한 번 → 날짜 드롭다운으로 비교, 자동 채점 표.

## 테스트: tests/test_journal.py (6개) — 파싱 견고성, 판정 로직, 채점 look-ahead 차단.

---

# (v8.3 zip 유지) 저널 표에 목표가·업사이드·핵심판단 추가
- JSON 스키마/통합 프롬프트에 target_price(목표가) 추가(target_pct·note는 기존).
- score_entry: 목표 업사이드%(목표가/진입가) 계산, 평가구간 고가가 목표가 도달 시
  목표달성(🎯) 판정. 업사이드는 평가 전(대기 상태)에도 표시.
- 표 컬럼 추가: 목표가 · 목표업사이드% · 목표달성(🎯) · 핵심판단(note).

---

# (v8.3 zip 유지) 미국 rate limit 캐싱 + 현금/매도 타이밍

## 미국 재무 rate limit — per-ticker 캐싱
- fundamental.py: 재무(.info, 야후가 가장 잘 막는 호출)를 종목별·데이터기준일별 캐시.
  성공분만 저장 → 재실행 시 야후 호출 0(차단 회피). 실패분만 다음에 재시도.
- 간격 1.2초+지터, rate limit 시 20초 백오프. 완전초기화에만 삭제(⚡ 점수재계산엔 보존).

## 시장국면 → 현금 비중 + 컷라인 (한국·미국 공통)
- engine: market_regime_score(0~100)를 df.attrs로 UI에 전달, 세션 저장(regime_us/kr).
- 리레이팅 탭 상단 배너: 국면 점수·신호(매수우위/선별/방어/현금우위)·권장 현금비중%.
- 메인 랭킹에 "💵 현금 컷라인" 삽입: 국면이 약할수록 컷라인이 위로 올라와
  현금보다 나은(매수 검토) 종목이 줄어듦. 컷라인 아래 = 신규 진입 보류 권장.

## 매도 타이밍 — 저널에 현재가 기준 신호
- journal: 진입 기록에 최신 종가 기준 신호 추가 — 🎯익절검토/🔴손절/🟠손절임박/🟠약세/🟢보유.
  평가창(N일)과 별개로 '지금' 보유 관리에 사용. 표에 현재가·현재수익%·매도신호 컬럼 추가.
