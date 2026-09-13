# AGENTS.md

## Project context

This is a Python/Streamlit stock screener for short-term stock/ETF candidates with 1-week to 2-month holding periods.

The project covers US and Korean markets. KOSPI, KOSDAQ, and US markets should not be treated as identical. Scoring behavior should remain configurable by market.

## Development rules

- Prefer small PR-sized changes.
- Do not make broad rewrites unless explicitly requested.
- Preserve existing behavior unless the task says otherwise.
- Before changing scoring, ranking, filtering, or backtesting logic, identify the expected impact.
- Do not add new dependencies without asking.
- Do not hardcode API keys, credentials, or local absolute paths.
- Keep data collection, scoring, ranking, backtesting, and UI responsibilities separated where practical.
- After each change, run the available tests or the smallest relevant smoke command.
- If tests cannot be run, explain why.

## Refactor priorities

1. Characterize current behavior before changing logic.
2. Reduce duplicated filter/exclusion logic.
3. Separate scoring concepts clearly: raw factor scores, final ranking score, validated/backtest score, and UI-only re-rating score.
4. Keep KOSPI, KOSDAQ, and US configuration separate.
5. Avoid changing ranking output and backtest logic in the same step unless explicitly requested.
