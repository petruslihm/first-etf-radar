# Implementation Status — refactor/screener-v9

Last updated: 2026-06-22  
Branch: `refactor/screener-v9`  
Tests: **230 / 230 passed**

---

## Completed Features

### Core (pre-existing, untouched)
| Feature | Files |
|---|---|
| US screener — scoring, ranking, backtest | `screener/factors.py`, `screener/scoring.py` |
| KR screener — scoring, ranking, backtest | `screener/factors.py`, `screener/scoring.py` |
| Shared filter helpers | `app.py` (`build_candidates`, filter utils) |
| Scoring contract docs | `docs/scoring_contract.md` |

### Added in this branch

#### US On-demand Ticker Analysis
- Fetch single US ticker via yfinance without disturbing loaded screener results.
- Runs full `calc_us_factors` → `validated_score` path.
- Session key: `ondemand_us_{ticker}`; never overwrites `us_full_df`.
- Entry point: `screener/collector.py::fetch_single_us_ticker_ondemand`

#### KR On-demand Ticker Analysis
- Fetch single KR ticker via Naver Finance scraping.
- Accepts 6-digit code (e.g. `005930`) or Korean name (e.g. `삼성전자`).
- Runs full `calc_kr_factors` path.
- Session key: `ondemand_kr_{query}`; never overwrites `kr_full_df`.
- Entry points: `screener/collector.py::fetch_single_kr_ticker_ondemand`, `_build_kr_offline_name_map`

#### KR DART Event Catalyst v1
- Module: `screener/events.py`
- Fetches recent disclosures from DART Open API (requires `DART_API_KEY`).
- Classifies filings: `positive` / `negative` / `neutral`.
- Scores: base 50, +10/pos (cap +25), −15/neg (cap −35), clamp 0–100.
- Returns `event_score`, `event_flags`, categorised events.
- Non-fatal: missing key → `event_score=None`, graceful error message.

#### US EDGAR Event Catalyst v1
- Module: `screener/events_us.py`
- Fetches recent SEC EDGAR filings via public submissions API (no API key required).
- Classifies by form type (S-3/424Bx → negative) and description keywords.
- Scores: base 50, +12/pos (cap +25), −18/neg (cap −35), 10-Q/K periodic +5, clamp 0–100.
- Returns `event_score`, `event_flags`, categorised events.
- Non-fatal: CIK not found or network error → `event_score=None`.

#### US Earnings Catalyst v1
- Function: `screener/events_us.py::fetch_us_earnings_catalyst`
- Reads `yfinance.Ticker.earnings_dates` for EPS Surprise(%).
- Scores: base 50, EPS >+10%→+15, >+3%→+8, <−5%→−12; revenue beat +8 / miss −8; dual beat +5; clamp 0–100.
- Non-fatal: no data → `earnings_score=None`.
- Revenue surprise not implemented in v1 (always `None`).

#### Event-adjusted Shadow Score
- Function: `app.py::_calc_event_adjusted_score`
- Combines `validated_score` + `event_score` into `event_adjusted_score`.
- Delta rules: event ≥75→+6, ≥65→+3, ≤35→−6, ≤25→−10; pos flags +2 (cap +8); risk flags −3 (floor −15); `top_risk_score` ≥85 & event <60 → −3; clamp 0–100.
- **Does NOT mutate `validated_score`, `final_score`, `rr_score`, `catalyst_score`.**
- Stored as `event_adjusted_score`, `event_delta`, `event_adjustment_reason` on on-demand result dicts only.

#### Shadow Event Score Comparison (UI)
- Collapsible expander at the bottom of the Score tab.
- Reads on-demand session cache for loaded screener top-20 candidates.
- Shows `validated_score`, `event_adj_score`, `delta`, `reason` in a table.
- No network calls; no ranking changes.

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `DART_API_KEY` | **Required** for KR DART event catalyst | DART Open API key — free signup at <https://opendart.fss.or.kr> |
| `SEC_USER_AGENT` | Optional | Override default SEC EDGAR `User-Agent` header (e.g. `"MyApp contact@example.com"`). Falls back to built-in default. |

No other secrets are hardcoded. All other data sources (Naver Finance, SEC EDGAR, yfinance) are either public APIs or scraping without auth.

---

## Manual Test Checklist

### US On-demand
- [ ] Enter `NVDA` in ticker lookup → click "⚡ 점수 계산" → see US on-demand card with `validated_score`
- [ ] Enter `AAPL` → see EDGAR event card + shadow score card + earnings catalyst card
- [ ] Enter `TSLA` → check event_delta reflects EDGAR signal direction
- [ ] Enter `XYZINVALID` → see error, loaded screener results **unchanged**
- [ ] Run GPT action → verify `stock_line` includes `EDGAR이벤트`, `이벤트조정점수`, `실적점수`

### KR On-demand
- [ ] Enter `005930` → see KR on-demand card with `validated_score`
- [ ] Enter `삼성전자` → resolves to `005930`, same result
- [ ] Enter `086520` → KOSDAQ bench used
- [ ] Enter `999999` (invalid) → error, loaded screener results **unchanged**
- [ ] With `DART_API_KEY` set: DART event card appears with score + flags
- [ ] Without `DART_API_KEY`: caption-only message, no crash

### DART / EDGAR
- [ ] DART: positive 잠정실적 공시 → `event_score` > 50
- [ ] DART: 유상증자 공시 → `event_score` < 50
- [ ] EDGAR: recent 8-K earnings beat → `event_score` > 50
- [ ] EDGAR: S-3 offering → `event_score` < 50
- [ ] Network failure → `event_score=None`, other analysis still renders

### Shadow Comparison
- [ ] Run on-demand for 2–3 tickers → open "⚡ Shadow Event Score Comparison" expander → table shows those tickers
- [ ] Before any on-demand run → expander shows "No event context loaded"
- [ ] Loaded screener ranking order **unchanged** after shadow section renders

---

## Known Limitations

| Area | Limitation |
|---|---|
| `event_adjusted_score` | Shadow score only — displayed for reference, not used in ranking or backtest |
| Default ranking / backtest | Completely unchanged; all new features are additive read-only layers |
| DART coverage | Only DART Open API disclosures; KIND (KRX) disclosures, proxy filings not included |
| EDGAR coverage | Recent filings in `submissions/CIK*.json` only; XBRL financials / 8-K exhibits not parsed |
| Earnings v1 | EPS surprise only (from `yfinance.earnings_dates`); revenue surprise always `None` |
| yfinance stability | `earnings_dates` column names may differ across yfinance versions; graceful fallback to `earnings_score=None` |
| KR name resolution | Name lookup uses `FORCED_KR_UNIVERSE` (~100 curated tickers); arbitrary names outside this list require 6-digit code input |
| Performance | EDGAR CIK lookup fetches `company_tickers.json` (~10 MB) on every on-demand call; no local cache |

---

## Test Coverage Summary

```
tests/test_ondemand.py      — 48 tests  (US on-demand, normalize, safe_row, shadow score)
tests/test_ondemand_kr.py   — 24 tests  (KR on-demand, name map, factors)
tests/test_kr_events.py     — 40 tests  (DART classify, score, fetch, integration)
tests/test_us_events.py     — 77 tests  (EDGAR classify, score, fetch, earnings catalyst)
tests/test_scoring.py       —  8 tests  (scoring formula regression)
tests/test_filters.py       — (existing)
Total                       — 230 passed / 0 failed
```

All tests are network-free (monkeypatch). No live API calls in CI.
