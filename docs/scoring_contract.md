# Scoring Contract

This document records the current scoring contract before any scoring, ranking,
filtering, or backtest refactor. It describes current behavior only. It should
not be read as a target design for the existing implementation.

The current implementation supports US and Korean markets, and Korean behavior
must continue to preserve KOSPI and KOSDAQ distinctions where the data path has
separate benchmark or universe handling. US, KOSPI, and KOSDAQ/KR scoring should
not be collapsed into a single undifferentiated market profile.

## Current Score Columns

### `leader_final`

1. Where it is calculated:
   - `screener/factors.py` in `calc_us_factors()`.
   - `screener/factors.py` in `calc_kr_factors()`.

2. Inputs it uses:
   - US path:
     - `leader_score`
     - `volume_score`
     - `breakout_score`
     - `catalyst_score`
     - `quality_score`
     - sector strength derived from sector relative behavior
     - US optimized weights from `screener.backtest.load_optimal_weights()`,
       with local defaults if loading fails
     - `us_haircut` from regime analysis
   - KR path:
     - `leader_score`
     - `volume_score`
     - `breakout_score`
     - `catalyst_score`
     - `quality_score`
     - sector strength
     - `supply_score`
     - KR optimized weights from `screener.backtest.load_optimal_weights_kr()`,
       with local defaults if loading fails
     - `kr_haircut` from regime analysis
   - Current behavior forces the quality weight to `0.0` in both US and KR
     factor paths before calculating `leader_final`, `buyable_final`, and
     `validated_score`.

3. Current usage:
   - Live ranking:
     - Used as a fallback when `validated_score` is unavailable.
     - Used indirectly because `final_score` is initially set to
       `leader_final`.
   - UI display:
     - Available in candidate row data and shown in candidate cards/detail
       surfaces.
   - Backtest:
     - Not the shared backtest score.
   - Explanation:
     - Acts as the current leader-oriented aggregate before later live overlays.

4. Market-specific future behavior:
   - Should remain market-specific.
   - US should keep US benchmark/sector assumptions separate from KR behavior.
   - KR should preserve KOSPI/KOSDAQ benchmark distinctions and supply-related
     inputs.

5. Known drift risks:
   - `leader_final` includes sector strength, KR supply, and regime haircut,
     while the shared validation path intentionally excludes some live overlays.
   - `final_score` starts from `leader_final` and can later diverge after engine
     overlays.
   - Optimized weight caches can alter behavior without code changes.
   - The quality input remains present in data but is neutralized by setting its
     weight to zero.

### `buyable_final`

1. Where it is calculated:
   - `screener/factors.py` in `calc_us_factors()`.
   - `screener/factors.py` in `calc_kr_factors()`.

2. Inputs it uses:
   - US path:
     - `momentum_score`
     - `buyable_score`
     - `volume_score`
     - `quality_score`
     - `catalyst_score`
     - sector strength
     - `top_risk_score`
     - US optimized/default weights
     - `us_haircut`
   - KR path:
     - `momentum_score`
     - `buyable_score`
     - `volume_score`
     - `supply_score`
     - `quality_score`
     - `catalyst_score`
     - sector strength
     - `top_risk_score`
     - KR optimized/default weights
     - `kr_haircut`
   - Quality is currently included as an input but effectively disabled through
     a zero weight.

3. Current usage:
   - Live ranking:
     - `screener/engine.py` uses `buyable_final` to rank the buyable track.
     - The buyable track is also gated by `buyable_score >= 55`.
   - UI display:
     - Available in row data.
   - Backtest:
     - Not the shared backtest score.
   - Explanation:
     - Represents the pullback/buyability aggregate rather than the leader
       aggregate.

4. Market-specific future behavior:
   - Should remain market-specific.
   - KR supply-related factors are asymmetric with US behavior and should remain
     explicit.

5. Known drift risks:
   - The buyable track combines a hard `buyable_score >= 55` gate with
     `buyable_final` sorting, so threshold behavior and weighted ranking can
     diverge.
   - `buyable_final` can disagree with `leader_final`, `final_score`, and
     `validated_score` by design.
   - KR includes supply behavior while US does not.
   - Quality is present but disabled.

### `final_score`

1. Where it is calculated:
   - Initially assigned in `screener/factors.py` in both `calc_us_factors()` and
     `calc_kr_factors()` as `leader_final`.
   - Mutated later in `screener/engine.py` by:
     - `_integrate_fundamentals()`
     - `_apply_theme_boost()`
     - `_apply_context_factors()`
     - `_apply_avoid_penalty()`

2. Inputs it uses:
   - Base value:
     - `leader_final`
   - Fundamental overlay:
     - `fund_score`
     - `fund_grade`
     - `rev_growth`
     - `op_growth`
     - `eps_growth`
     - `rev_accel`
     - related stored fundamental metadata
   - Theme overlay:
     - manually or AI-selected theme tickers
     - theme role/weight metadata
     - strongest 20-day theme member boost
   - Context overlay:
     - context multipliers for catalyst, momentum, volume, supply, and buyable
   - Avoid overlay:
     - avoid ticker list
     - risk score increment
     - fixed `final_score` penalty

3. Current usage:
   - Live ranking:
     - Selectable in the UI sort.
     - Not the default main candidate sort.
   - UI display:
     - Displayed in reports and fundamentals views as the adjusted aggregate
       score.
   - Backtest:
     - Not the shared backtest-aligned score.
   - Explanation:
     - Represents post-factor live overlays and should be interpreted as mutable
       live state.

4. Market-specific future behavior:
   - Should remain market-specific.
   - KR supply and KOSPI/KOSDAQ context should not be folded into US behavior.
   - US analyst/fundamental availability is asymmetric with KR and should remain
     explicit where it affects future score structure.

5. Known drift risks:
   - `final_score` is mutable after factor calculation.
   - `validated_score` is the backtest-aligned/shared validation path, while
     `final_score` receives live-only overlays.
   - Fundamental integration updates `final_score` and display metadata without
     recomputing `leader_final` or `validated_score`.
   - Theme/context/avoid overlays can change `final_score` without changing the
     backtest validation score.
   - UI reports may show `final_score` while the main candidate default sort uses
     `validated_score`.

### `validated_score`

1. Where it is calculated:
   - Shared formula in `utils/scoring.py`:
     - `calc_validated_score()`
     - `calc_validated_score_series()`
   - Assigned to live factor rows in `screener/factors.py`:
     - `calc_us_factors()`
     - `calc_kr_factors()`
   - Used by backtest paths in `screener/backtest.py`, where `composite` and
     `validated_score` are treated as the same shared formula.

2. Inputs it uses:
   - Base formula inputs:
     - `leader`
     - `volume`
     - `breakout`
     - `catalyst`
     - `quality`
     - `risk`
     - `ret_20d`
   - Series formula enhanced inputs when present:
     - `volume_enhanced` over `volume`
     - `breakout_enhanced` over `breakout`
     - `risk_combined` over `risk`
   - Series formula optional newer factors when weighted:
     - `momentum_accel`
     - `near_high`
     - `vol_contract`
     - `pullback_q`
     - `brk_persist`
     - `sect_rs`
     - `supply_flow`
   - Risk penalty:
     - calculated from `risk` and `ret_20d`
   - US live path:
     - mostly passes base factor columns into the shared formula.
   - KR live path:
     - passes enhanced and newer KR factor columns, including supply-flow style
       inputs where available.

3. Current usage:
   - Live ranking:
     - Default candidate sort in `app.py`.
     - Used by `screener/engine.py` when constructing the leader pool.
     - Used as the preferred ordering for selecting candidates for fundamental
       collection.
   - UI display:
     - Displayed as the backtest validation score.
   - Backtest:
     - Main shared backtest-aligned score.
     - Backtest output also records `composite` as the same concept in several
       paths.
   - Explanation:
     - Explains the historically validated factor blend better than `final_score`
       or `rr_score`.

4. Market-specific future behavior:
   - Should remain market-specific through weights and enabled factor sets.
   - The shared scoring function can remain common, but US, KOSPI, and KOSDAQ/KR
     should have explicit configuration boundaries.

5. Known drift risks:
   - The scalar and series functions do not have identical breadth: the series
     path supports enhanced and newer factor columns.
   - Live US and live KR pass different factor sets.
   - KR includes supply-related factor availability that US does not.
   - `composite` duplicates `validated_score` in backtest output and can obscure
     which name is authoritative.
   - Optimized weight caches can change `validated_score` behavior independently
     of source changes.
   - `final_score` overlays are not part of `validated_score`, so live display
     and validation ranking can diverge.

### `rr_score`

1. Where it is calculated:
   - `app.py` in `calc_rerating_score()`.
   - Called from candidate lookup/building paths including `build_candidates()`.

2. Inputs it uses:
   - Row inputs:
     - `is_theme_pick`
     - `catalyst_score`
     - `rs_rank_pct`
     - `ret_20d`
     - `breakout_score`
     - `volume_score`
     - `top_risk_score`
     - `ret_5d`
     - `risk_flag`
   - Fundamental inputs:
     - `rev_accel`
     - `fund_score`
     - `rev_growth`
   - Analyst inputs:
     - `eps_revision`
     - `eps_trend`
     - `beat_count`
     - `analyst_score` is read into a local variable but is not currently used
       in the score calculation.

3. Current usage:
   - Live ranking:
     - Optional UI sort key.
   - UI display:
     - Prominently displayed on candidate cards and detail surfaces.
   - Backtest:
     - Not used by backtest.
   - Explanation:
     - UI-layer, rerating-oriented score for narrative and candidate review.

4. Market-specific future behavior:
   - Should become explicitly market-specific if retained.
   - US analyst data is available through the US analyst collection path.
   - KR currently passes an empty analyst map in candidate building, so KR and US
     rerating inputs are asymmetric.

5. Known drift risks:
   - Calculated in the UI layer rather than the scoring engine.
   - Not part of the backtest validation path.
   - Can rank candidates differently from `validated_score` and `final_score`.
   - Analyst and fundamental availability is asymmetric between US and KR.
   - Risk logic depends partly on `risk_flag` text, which is fragile as a scoring
     dependency.

## Target Concepts Not Yet Separate

### `core_score`

1. Where it is calculated:
   - Not currently present as a separate repository column or function output.

2. Inputs it uses:
   - Not applicable in the current implementation.
   - The closest existing concepts are the raw factor blends inside
     `leader_final`, `buyable_final`, and `validated_score`.

3. Current usage:
   - Live ranking: not used as a separate score.
   - UI display: not displayed as a separate score.
   - Backtest: not used as a separate score.
   - Explanation: only implicit through existing aggregate scores.

4. Market-specific future behavior:
   - Should be market-specific if introduced.
   - US, KOSPI, and KOSDAQ/KR should have configurable factor sets and weights.

5. Known drift risks:
   - Without a separate `core_score`, technical strength is mixed into
     `leader_final`, `validated_score`, and mutable `final_score`.
   - It is hard to explain raw technical strength separately from overlays.

### `event_score`

1. Where it is calculated:
   - Not currently present as a separate repository column or function output.

2. Inputs it uses:
   - Not applicable in the current implementation.
   - Closest existing inputs are `catalyst_score`, theme boosts, fundamental
     acceleration, earnings/analyst inputs, and context factors.

3. Current usage:
   - Live ranking: not used as a separate score.
   - UI display: not displayed as a separate score.
   - Backtest: not used as a separate score.
   - Explanation: event-like signals are currently spread across
     `catalyst_score`, `final_score`, and `rr_score`.

4. Market-specific future behavior:
   - Should be market-specific if introduced.
   - US analyst/fundamental availability and KR supply/disclosure behavior are
     asymmetric and should not be forced into the same event model.

5. Known drift risks:
   - Event-like signals affect different scores in different layers.
   - Theme and context boosts mutate `final_score`.
   - Analyst revisions affect `rr_score`, not `validated_score`.
   - Fundamental acceleration can adjust display catalyst metadata and
     `final_score` without recomputing `validated_score`.

### `overlay_score`

1. Where it is calculated:
   - Not currently present as a separate repository column or function output.

2. Inputs it uses:
   - Not applicable in the current implementation.
   - The closest current concept is the implicit set of post-factor adjustments
     applied to `final_score`.

3. Current usage:
   - Live ranking: not used as a separate score.
   - UI display: not displayed as a separate score.
   - Backtest: not used as a separate score.
   - Explanation: implicit inside mutable `final_score`.

4. Market-specific future behavior:
   - Should be market-specific if introduced.
   - Overlay policy should keep US, KOSPI, and KOSDAQ/KR behavior configurable.

5. Known drift risks:
   - Current overlays are destructive mutations of `final_score`.
   - The system cannot show how much of `final_score` came from raw factor
     strength versus live overlays.
   - Overlay impact is not isolated in backtest.

## Proposed Future Target Structure

### Raw factor columns

Raw factor columns should remain the lowest-level observed or derived signals.
They should not include UI-only rerating or live-only post-hoc boosts.

Examples from the current repository:

- `leader_score`
- `momentum_score`
- `volume_score`
- `buyable_score`
- `breakout_score`
- `top_risk_score`
- `quality_score`
- `catalyst_score`
- `supply_score`
- `volume_enhanced`
- `breakout_enhanced`
- `risk_combined`
- `momentum_accel`
- `near_high`
- `vol_contract`
- `pullback_q`
- `brk_persist`
- `sect_rs`
- `supply_flow`

### `core_score`

Market-configured blend of raw technical factors before event overlays and
final live adjustments.

Expected role:

- intermediate score
- explanation aid
- backtest feature analysis
- market-specific technical strength measure

### `event_score`

Market-configured blend of catalyst and event-like signals.

Expected role:

- isolate catalyst/theme/earnings/analyst effects
- preserve US and KR asymmetry
- keep event-driven upside separate from core technical strength

### `risk_adjusted_score`

Risk-adjusted version of the core/event score before mutable live overlays.

Expected role:

- apply validated risk penalties
- support backtest comparison
- explain why high raw strength is discounted

### `final_score`

Live adjusted score after explicit overlays.

Expected role:

- live ranking only when overlays are intentionally part of the strategy
- UI display as the current adjusted score
- clear decomposition into base score plus overlay components

### `validated_score`

Backtest-aligned score using only historically available and validated inputs.

Expected role:

- default validation metric
- backtest and live shared ranking path
- score monitoring
- market-specific weights and enabled factors

### `ui_rr_score`

Renamed and isolated UI-layer rerating score.

Expected role:

- candidate narrative
- optional UI sort
- analyst/fundamental rerating view
- not used by backtest unless promoted into an explicitly validated scoring path
