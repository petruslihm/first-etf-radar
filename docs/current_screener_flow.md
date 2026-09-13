# Current Screener Flow

This document describes the implementation as it exists now. It is not a target
architecture and does not propose formula changes.

## 1. Data flow

### Universe collection

The Streamlit entry points are `run_us_screener()` and `run_kr_screener()` in
`app.py`.

US universe collection starts in `screener.collector.get_us_tickers()`:

- Reads current S&P 500 and Nasdaq-100 constituents from Wikipedia.
- Falls back to `data.us_universe.SP500_FALLBACK` when the S&P scrape is weak.
- Adds `screener.universe.US_BENCH` benchmark ETFs and `UNIVERSE_ETFS`.
- Adds `data.us_universe.FORCED_UNIVERSE`.
- Adds a dynamic momentum universe from `_fetch_us_momentum_universe()`.
- Removes hardcoded dead/delisted/problem tickers.
- Caches the ticker list under the data-date cache key.

KR universe collection starts in `screener.collector.get_kr_tickers()`:

- Collects market-cap lists from Naver for KOSPI 200 and KOSDAQ 150.
- Adds top trading-value names from Naver.
- Adds `data.kr_universe.FORCED_KR_UNIVERSE`.
- Returns one combined KR ticker list.

KOSPI and KOSDAQ are tagged on each KR item during `fetch_kr_universe()` using
membership in the collected KOSPI/KOSDAQ sets, but the main KR screener pipeline
still processes one combined KR universe.

### OHLCV, benchmark, supply, fundamental, and theme collection

US OHLCV and metadata are collected by `fetch_us_universe()`:

- Uses `yfinance.download()` in batches for OHLCV.
- Fetches per-ticker metadata through `yfinance.Ticker.info` or `fast_info`.
- Stores fields used later by filters and factors: sector, market cap, price,
  average dollar volume, company name, and OHLCV.
- Benchmark and sector ETF OHLCV are present because benchmark ETFs are included
  in the US ticker universe. `run_us_pipeline()` also calls
  `screener.backtest.fetch_sector_etfs()` to fill missing sector ETF series.

KR OHLCV and metadata are collected by `fetch_kr_universe()`:

- Uses Naver day-price pages via `_naver_ohlcv()`.
- Fetches recent foreign/institution supply aggregates via `_naver_supply()`.
- Fetches market cap from Naver item pages via `_naver_marcap_single()`.
- Adds name, KOSPI/KOSDAQ market label, price, average amount, and OHLCV.
- `fetch_kr_benchmarks()` separately collects KOSPI and KOSDAQ index series
  from Naver, with yfinance ETF fallback using `069500.KS` and `229200.KS`.
- Historical KR supply series for backtests are handled separately by
  `attach_kr_supply_series()` and `_naver_supply_series()`.

Fundamentals are integrated inside `screener.engine._integrate_fundamentals()`:

- It selects only a pre-ranked subset, using `validated_score` when available
  and `leader_final` otherwise.
- US fundamentals come from yfinance in `fetch_us_fundamental()`.
- KR fundamentals come from FnGuide in `fetch_kr_fundamental()`.
- `fetch_fundamentals_batch()` caches by market/ticker/date and rate-limits.
- Fundamental collection updates metadata columns and adjusts `final_score` only;
  it does not recompute `leader_final` or `validated_score`.

US analyst data is collected after the US pipeline in `run_us_screener()`:

- Candidate targets are taken from leader, breakout, turnaround, buyable, and
  theme lists, deduped and capped.
- `fetch_analyst_batch()` populates `st.session_state["us_analyst"]`.
- KR does not have a parallel analyst collection path in the current app.

Theme inputs are stored in session state by the advanced/theme UI. During the
pipeline, `_apply_theme_boost()` marks theme picks and adds boosts to
`final_score`. `calc_universe_theme_strength()` and `calc_theme_score()` produce
theme summaries for display.

### Factor calculation

Raw factor functions live in `screener.factors`:

- Shared raw factors include `leader_score`, `momentum_score`, `volume_score`,
  `buyable_score`, `breakout_score`, `top_risk_score`,
  `quality_of_trend_score`, and `catalyst_proxy_score`.
- Shared helper `_core_factor_scores()` is used by both US and KR row builders.
- US row construction happens in `calc_us_factors()`.
- KR row construction happens in `calc_kr_factors()`.

US factor rows compare stocks to SPY, QQQ, and sector benchmarks. They compute
sector-relative values and market-regime haircut, then produce raw factor
columns, `leader_final`, `buyable_final`, `final_score`, `track`, and
`validated_score`.

KR factor rows compare each stock to the benchmark implied by its `market` field:
KOSPI stocks use `kospi`, KOSDAQ stocks use `kosdaq`. KR also computes
`supply_score`, newer rapid-move factors such as `momentum_accel`,
`volume_shock`, `brk_persist`, `near_high`, `vol_contract`, `pullback_q`,
`risk_filter`, plus `sect_rs`, `supply_flow`, and enhanced volume/breakout/risk
inputs.

### Scoring

Row-level live scoring is split across `screener.factors`,
`utils.scoring`, and `screener.engine`:

- `leader_final` and `buyable_final` are calculated in `calc_us_factors()` and
  `calc_kr_factors()`.
- `final_score` initially equals `leader_final`.
- `validated_score` is calculated from `utils.scoring.calc_validated_score_series()`
  or `calc_validated_score()`.
- `_integrate_fundamentals()`, `_apply_theme_boost()`,
  `_apply_context_factors()`, and `_apply_avoid_penalty()` adjust
  `final_score` after row construction.
- The UI computes `rr_score` separately in `app.calc_rerating_score()`.

### Track classification

Track labels are produced by `screener.factors.classify_track()` from raw
`leader_score`, `buyable_score`, `breakout_score`, `top_risk_score`, and
`momentum_score`.

Current labels include:

- `Hot Leader / Extended`
- `Hot Leader / Buyable`
- `Breakout Signal`
- `Leader / Buyable`
- `Leader / Pullback Wait`
- `Watch Only`
- `Avoid / Weak`

`screener.engine._split_tracks()` then builds four output buckets:

- Leader: predefined leader tracks, plus top `validated_score` names merged in.
- Breakout: `Breakout Signal`, sorted by `breakout_score`.
- Buyable: rows with `buyable_score >= 55`, sorted by `buyable_final`.
- Turnaround: derived in `_split_tracks()` from `week52_low_prox`,
  `catalyst_score`, and `momentum_score`, then labelled
  `Turnaround / Early`.

### Ranking and splitting

The engine returns:

`leader_df, buyable_df, breakout_df, turnaround_df, excluded, full_df`

for both US and KR.

The Streamlit UI stores these in market-specific session keys. The main
candidate UI does not display the engine buckets directly. Instead,
`app.build_candidates()` merges `turnaround`, `leader`, `breakout`, and
`buyable`, removes duplicates, applies UI-level filters, computes `rr_score`,
and sorts by the selected key. The default sort key is `validated_score`.

### Streamlit display

`app.main()` sets the active market and renders tabs. The primary candidate tab
uses `render_rerating_tab()`:

- Builds candidates with `build_candidates()`.
- Shows summary stats, cash guidance, saved journal rows, and candidate cards.
- Candidate cards use `rr_score` prominently and also show `validated_score`,
  `leader_final`, raw factors, trend metrics, fundamentals, analyst data when
  present, and links.

The GPT analysis tab also calls `build_candidates()` and can sort by several
ad-hoc modes. Quick ticker lookup first searches merged candidates, then falls
back to `*_full_df`.

### Report and backtest output

CSV/chart/report helpers live in `screener.reporter`:

- `save_chart()` writes per-ticker chart images.
- `save_csv()` writes CSV output under `reports`.
- `save_html_report()` can render grouped US/KR HTML reports and displays
  `final_score`, not `validated_score` or `rr_score`, in the score column.

Backtest output lives in `screener.backtest`:

- US uses `run_walkforward_backtest()` and optional `run_snapshot_backtest()`.
- KR uses `run_kr_walkforward_backtest()`.
- Both write records with `composite` and `validated_score`.
- Backtest results are cached under `CACHE_DIR`; the Streamlit advanced tab
  displays result summaries, factor diagnostics, and weight optimization UI.
- Journal/report-like pick tracking is handled separately in `screener.journal`
  from saved GPT JSON picks and later OHLCV evaluation.

## 2. Scoring concepts

### `leader_final`

Calculated in:

- `screener.factors.calc_us_factors()`
- `screener.factors.calc_kr_factors()`

Meaning in current code: a live row-level leader/ranking score assembled from
raw leadership, volume, breakout, catalyst, sector strength, and market-regime
haircut. KR also includes supply. The code sets `final_score` equal to
`leader_final` before later engine adjustments.

### `buyable_final`

Calculated in:

- `screener.factors.calc_us_factors()`
- `screener.factors.calc_kr_factors()`

Meaning in current code: a row-level score for nearer-term entry quality. It
weights `buyable_score` more strongly than `leader_final`, includes momentum,
volume, catalyst, sector strength, risk penalty, and for KR, supply. It is used
by `_split_tracks()` to rank buyable candidates.

### `final_score`

Initially set in:

- `screener.factors.calc_us_factors()`
- `screener.factors.calc_kr_factors()`

Adjusted in:

- `screener.engine._integrate_fundamentals()`
- `screener.engine._apply_theme_boost()`
- `screener.engine._apply_context_factors()`
- `screener.engine._apply_avoid_penalty()`

Meaning in current code: the mutable live/UI composite score. It begins as
`leader_final`, then receives non-backtest adjustments such as fundamentals,
theme boosts, context multipliers, and AI avoid penalties. It is displayed in
reports and can be selected as a Streamlit sort key.

### `validated_score`

Calculated in:

- `screener.factors.calc_us_factors()`
- `screener.factors.calc_kr_factors()`
- `screener.backtest.run_walkforward_backtest()`
- `screener.backtest.run_kr_walkforward_backtest()`
- Several optimization and diagnostic helpers through
  `utils.scoring.calc_validated_score_series()`

Meaning in current code: the score intended to match the backtest-validated
formula. The base function is `utils.scoring.calc_validated_score_series()`,
which normalizes configured positive weights and subtracts risk penalty. Live US
passes basic raw factors. Live KR passes enhanced and newer factor columns.
Backtests call the same function but may supply neutral defaults for unavailable
sector/supply inputs.

### `rr_score`

Calculated in:

- `app.calc_rerating_score()`
- `app.build_candidates()`
- `app._find_ticker_candidate()` fallback path

Meaning in current code: a UI-only rerating/GPT candidate score. It starts at 50
and adjusts for analyst EPS revisions and beat history, fundamentals, theme and
catalyst signals, relative strength, 20-day return, breakout/volume, and a small
risk penalty. US can use analyst data; KR currently uses fundamentals and row
data but an empty analyst dict.

## 3. Market differences

US and KR are separate pipelines at the app and engine level:

- US runner: `run_us_screener()` -> `get_us_tickers()` ->
  `fetch_us_universe()` -> `run_us_pipeline()`.
- KR runner: `run_kr_screener()` -> `get_kr_tickers()` ->
  `fetch_kr_universe()` plus `fetch_kr_benchmarks()` ->
  `run_kr_pipeline()`.

Current US behavior:

- Universe comes from S&P 500, Nasdaq-100, fallback, forced tickers, dynamic
  momentum tickers, and benchmark/sector ETFs.
- OHLCV and metadata come from yfinance.
- Benchmarks are SPY/QQQ plus sector ETFs.
- Relative strength is primarily against SPY and sector ETFs.
- Sector normalization uses GICS-like yfinance sector names.
- Fundamentals and analyst data both use yfinance.
- Live filtering uses price, market cap, average dollar volume, history length,
  and explicit benchmark/ETF exclusion from the candidate universe.

Current KR behavior:

- Universe comes from Naver KOSPI/KOSDAQ market-cap lists, trading-value lists,
  and forced KR tickers.
- OHLCV, supply, market cap, warnings, sectors, and benchmarks mostly come from
  Naver. Fundamentals come from FnGuide.
- Benchmarks are KOSPI and KOSDAQ series.
- Each KR row chooses the benchmark from its `market` field for relative
  strength, and KR backtest records both KOSPI/KOSDAQ excess returns.
- KR scoring includes `supply_score` and additional enhanced factors more
  prominently than US.
- KR filtering includes preferred-share-like code filtering, warned stock
  filtering, ETF/ETN name filtering, market cap, average amount, and history.

KOSPI and KOSDAQ are not fully separated as independent configurable pipelines.
They are labelled and benchmarked differently inside the combined KR pipeline,
but they share one KR universe collection result, one KR filter config
(`KR_FILTER`), one `run_kr_pipeline()`, one KR factor formula path, one KR
session-state set, and one KR optimized-weight cache.

## 4. Filtering and exclusion

Filtering/exclusion currently happens in several places.

Universe collection:

- `get_us_tickers()` removes hardcoded dead/problem US tickers.
- `get_us_tickers()` includes benchmark/sector ETFs intentionally so they are
  available as benchmark data.

Live US engine:

- `run_us_pipeline()` builds `all_etf_set` from `US_BENCH`, `UNIVERSE_ETFS`,
  and additional sector ETFs, then excludes those from individual-stock scoring.
- It excludes rows below `US_FILTER` thresholds for price, market cap, average
  dollar volume, and history length.

Live KR engine:

- `run_kr_pipeline()` excludes preferred-share-like numeric codes when
  `KR_FILTER["exclude_preferred"]` is true.
- It excludes names matching an inline `KR_ETF_KEYWORDS` set.
- It excludes Naver warned/caution/management names and `item["is_warned"]`.
- It excludes rows below `KR_FILTER` thresholds for market cap, average amount,
  and history length.

UI candidate merge:

- `app.build_candidates()` filters by track labels.
- It can exclude turnaround candidates from the main list.
- It repeats KR ETF-name exclusion with another keyword set.
- It excludes very high-risk rows only under a narrower condition:
  `risk >= 90`, negative 5-day return, and a bearish risk flag.

US analyst target collection:

- `run_us_screener()` excludes `screener.backtest.EXCLUDE_BT_TICKERS` from the
  analyst collection target list.

US backtest:

- `run_walkforward_backtest()` excludes `EXCLUDE_BT_TICKERS`.
- It also requires OHLCV and minimum history.

KR backtest:

- `run_kr_screener()` has a precheck using `EXCLUDE_KR_BT_TICKERS`,
  `KR_ETF_NAME_KEYWORDS`, benchmark codes, minimum price, minimum amount, and
  minimum history.
- `run_kr_walkforward_backtest()` repeats and expands this with
  `EXCLUDE_KR_BT_TICKERS`, `KR_BENCH_CODES`, `_is_kr_unwanted_security()`,
  `_kr_has_suspicious_price_action()`, low-price checks, low-liquidity checks,
  and return-outlier filters.

Duplicated logic is present:

- KR ETF/security exclusion keywords exist in the live engine, UI candidate
  merge, KR backtest precheck, and KR backtest proper.
- KR low price/liquidity/history checks exist in both app-level KR backtest
  precheck and `run_kr_walkforward_backtest()`.
- US ETF/benchmark exclusion appears in `run_us_pipeline()`,
  `EXCLUDE_BT_TICKERS`, and the US analyst target filter.
- Risk-based UI exclusion is separate from engine track exclusion and backtest
  filtering.

## 5. Backtest/live drift risks

Known places where live scoring, UI scoring, and backtest scoring may diverge:

- `final_score` receives fundamentals, theme boosts, context factors, and avoid
  penalties in the live engine; `validated_score` and most backtest composites
  do not include those same mutable adjustments.
- `rr_score` is calculated only in `app.py` and is not part of engine or
  backtest scoring.
- `leader_final` and `buyable_final` use formula blocks in `calc_us_factors()`
  and `calc_kr_factors()`, while backtest records store `composite` from
  `calc_validated_score_series()`.
- Live US validated scoring passes mostly base factors. Live KR validated
  scoring passes enhanced and newer KR factors. Backtest `_calc_scores_at()`
  supplies generic enhanced factors but uses neutral `sect_rs` and `supply_flow`
  unless KR historical supply series is attached.
- Backtest `_calc_scores_at()` does not use the full live row builders
  `calc_us_factors()` or `calc_kr_factors()`, so sector averages, regime
  details, and KR market-specific row context can differ.
- KR live scoring uses recent `_naver_supply()` aggregates. KR backtest can use
  historical `supply_series`, but only when it has been attached; otherwise
  supply flow stays neutral.
- Live KR filters and KR backtest filters are similar but not identical.
- KOSPI/KOSDAQ labels affect benchmark choice, but KR optimized weights and KR
  live scoring are still shared across the combined KR pipeline.
- Cached factor data can be reused and then have theme/context/avoid adjustments
  applied on top, so repeated runs with changed UI context can produce adjusted
  `final_score` without recomputing all raw factors.
- US analyst data is collected after the US engine run and affects `rr_score`,
  not the engine-produced `final_score` or `validated_score`.
- `save_html_report()` displays `final_score`, while the main candidate default
  sort uses `validated_score` and cards emphasize `rr_score`.
- Weight caches are separate for US and KR, but stale optimized weights can
  influence live `leader_final`, `buyable_final`, and `validated_score` until
  caches are cleared or overwritten.

## 6. Safe next refactor candidates

Small, low-risk next steps:

- Extract KR ETF/security exclusion keywords into one shared helper and use it
  from live engine, UI candidate merge, and KR backtest.
- Add a small documentation or characterization test for the current meanings of
  `leader_final`, `buyable_final`, `final_score`, `validated_score`, and
  `rr_score`.
- Create a single candidate-merge helper that accepts already-built engine
  buckets and applies UI filters in one place.
- Separate KOSPI and KOSDAQ configuration objects before changing any formulas.
- Add a lightweight comparison report that shows, for the same rows,
  `leader_final`, `final_score`, `validated_score`, and `rr_score` side by side
  to make future scoring drift visible.
