"""
screener/event_shadow.py
Top-N Batch Shadow Event Enrichment — pure, dependency-injectable helpers.

SHADOW-ONLY. Nothing here mutates validated_score / final_score / rr_score /
catalyst_score, changes candidate selection, ranking, or backtest. These helpers
only *read* candidate rows and produce a parallel "shadow" view that the UI can
display for inspection / later A-B analysis.

No network calls happen at import time or on page render — only when a fetcher is
actually invoked (batch button). All fetchers are dependency-injected so tests can
run fully offline.
"""
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Local cache for batch shadow rows (gitignored).
DEFAULT_CACHE_PATH = Path(__file__).parent.parent / ".cache" / "event_shadow_cache.json"


# ─── small utilities ─────────────────────────────────────────────────


def _to_float_or_none(v) -> "float | None":
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if isinstance(f, float) and math.isnan(f):
        return None
    return f


def _row_get(row, key, default=None):
    """Defensive read from a pandas Series or dict-like row."""
    if row is None:
        return default
    try:
        if hasattr(row, "get"):
            v = row.get(key, default)
        else:
            v = row[key]
    except Exception:
        return default
    return default if v is None else v


def _source_state(d: dict, score_key: str) -> str:
    if not isinstance(d, dict):
        return "unavailable"
    if d.get(score_key) is not None:
        return "ok"
    if d.get("error"):
        return "error"
    return "missing"


def _flatten_error(err) -> "str | None":
    if err is None:
        return None
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        parts = [f"{k}: {v}" for k, v in err.items() if v]
        return " | ".join(parts) if parts else None
    return str(err)


# ─── 1. pure context combiners ───────────────────────────────────────


def combine_us_event_context(edgar_context, earnings_context) -> dict:
    """
    Combine EDGAR event context + earnings catalyst context into one normalized
    shadow context. Never raises on missing keys / None / malformed input.

    Combined event_score:
      both present  → round(0.60*edgar + 0.40*earnings)
      edgar only    → edgar
      earnings only → earnings
      neither       → None
    """
    edgar = edgar_context if isinstance(edgar_context, dict) else {}
    earn  = earnings_context if isinstance(earnings_context, dict) else {}

    edgar_score    = _to_float_or_none(edgar.get("event_score"))
    earnings_score = _to_float_or_none(earn.get("earnings_score"))

    if edgar_score is not None and earnings_score is not None:
        combined = round(0.60 * edgar_score + 0.40 * earnings_score)
    elif edgar_score is not None:
        combined = round(edgar_score)
    elif earnings_score is not None:
        combined = round(earnings_score)
    else:
        combined = None

    flags = list(edgar.get("event_flags") or []) + list(earn.get("earnings_flags") or [])

    errors: dict = {}
    if edgar.get("error"):
        errors["edgar"] = edgar.get("error")
    if earn.get("error"):
        errors["earnings"] = earn.get("error")

    summary_parts = []
    if edgar_score is not None:
        summary_parts.append(f"EDGAR {edgar_score:.0f}")
    if earnings_score is not None:
        summary_parts.append(f"Earnings {earnings_score:.0f}")
    event_summary = " · ".join(summary_parts) if summary_parts else "이벤트 없음"

    return {
        "event_score":    combined,
        "edgar_score":    edgar_score,
        "earnings_score": earnings_score,
        "event_flags":    flags,
        "event_summary":  event_summary,
        "source_status":  {
            "edgar":    _source_state(edgar, "event_score"),
            "earnings": _source_state(earn, "earnings_score"),
        },
        "error":          errors or None,
    }


def normalize_kr_event_context(dart_context) -> dict:
    """
    Normalize a DART event context into the same shadow-context shape used by the
    US combiner. Preserves watch/positive/negative/neutral event lists.
    Never raises on missing keys / no API key / network-failure context / malformed.
    """
    d = dart_context if isinstance(dart_context, dict) else {}
    dart_score = _to_float_or_none(d.get("event_score"))
    error      = d.get("error")
    status     = _source_state(d, "event_score")

    if dart_score is not None:
        summary = f"DART {dart_score:.0f}"
    elif error and ("API 키" in str(error) or "DART_API_KEY" in str(error)):
        summary = "DART 키 없음"
    else:
        summary = "이벤트 없음"

    return {
        "event_score":     dart_score,
        "dart_score":      dart_score,
        "event_flags":     list(d.get("event_flags") or []),
        "event_summary":   summary,
        "source_status":   {"dart": status},
        "error":           {"dart": error} if error else None,
        "watch_events":    list(d.get("watch_events") or []),
        "positive_events": list(d.get("positive_events") or []),
        "negative_events": list(d.get("negative_events") or []),
        "neutral_events":  list(d.get("neutral_events") or []),
    }


# ─── 2. shadow-adjusted score (canonical) ────────────────────────────


def calc_event_adjusted_shadow_score(base_score, event_context, row=None) -> dict:
    """
    Canonical event-adjusted shadow score. Reusable pure function; mirrors the
    delta logic that used to live in app.py::_calc_event_adjusted_score.

    Never mutates `row`. Clamps to 0-100. If there is no usable event score, the
    base score is returned unchanged with delta 0 and a clear reason (so batch
    ranking stays stable for un-enriched rows).

    Returns: event_adjusted_score, event_delta, event_adjustment_reason,
             event_flags, event_risk_flags
    """
    try:
        base_f = float(base_score)
    except (TypeError, ValueError):
        base_f = 0.0

    risk_score = _row_get(row, "top_risk_score") if row is not None else None

    ev = event_context if isinstance(event_context, dict) else {}
    ev_score = _to_float_or_none(ev.get("event_score"))

    if ev_score is None:
        return {
            "event_adjusted_score":    round(base_f, 1),
            "event_delta":             0.0,
            "event_adjustment_reason": "이벤트 데이터 없음 (기준점수 유지)",
            "event_flags":             [],
            "event_risk_flags":        [],
        }

    # ── base delta from event_score (preserved logic) ──
    if ev_score >= 75:
        delta = 6.0;   reason_base = f"이벤트강세({ev_score:.0f})"
    elif ev_score >= 65:
        delta = 3.0;   reason_base = f"이벤트양호({ev_score:.0f})"
    elif ev_score <= 25:
        delta = -10.0; reason_base = f"이벤트위험({ev_score:.0f})"
    elif ev_score <= 35:
        delta = -6.0;  reason_base = f"이벤트부정({ev_score:.0f})"
    else:
        delta = 0.0;   reason_base = f"이벤트중립({ev_score:.0f})"

    # ── flag-based micro adjust (✅ / ⚠ only, to preserve existing behavior) ──
    all_flags  = ev.get("event_flags") or []
    pos_flags  = [f for f in all_flags if str(f).startswith("✅")]
    risk_flags = [f for f in all_flags if str(f).startswith("⚠")]

    if pos_flags:
        delta = min(delta + 2.0, 8.0)    # cap +8
    if risk_flags:
        delta = max(delta - 3.0, -15.0)  # floor -15

    reasons = [reason_base]
    if risk_score is not None:
        try:
            if float(risk_score) >= 85 and ev_score < 60:
                delta = max(delta - 3.0, -15.0)
                reasons.append("고위험종목")
        except (TypeError, ValueError):
            pass

    if pos_flags:
        reasons.append(f"긍정공시+{len(pos_flags)}")
    if risk_flags:
        reasons.append(f"부정공시-{len(risk_flags)}")

    adj = max(0.0, min(100.0, base_f + delta))
    return {
        "event_adjusted_score":    round(adj, 1),
        "event_delta":             round(delta, 1),
        "event_adjustment_reason": " / ".join(reasons),
        "event_flags":             pos_flags,
        "event_risk_flags":        risk_flags,
    }


# ─── KR market-segment tag (no scoring change, for later A-B only) ────


def infer_kr_market_segment(row) -> str:
    """
    Best-effort KOSPI / KOSDAQ / UNKNOWN tag from existing row fields only.
    No network calls. Returns "UNKNOWN" when undeterminable.
    """
    fields = []
    for key in ("market", "exchange", "benchmark", "bench_name", "segment", "market_segment"):
        v = _row_get(row, key)
        if v:
            fields.append(str(v))
    tk = _row_get(row, "ticker")
    if tk:
        fields.append(str(tk))

    blob = " ".join(fields).upper()
    if "KOSDAQ" in blob or ".KQ" in blob or "코스닥" in blob:
        return "KOSDAQ"
    if "KOSPI" in blob or ".KS" in blob or "코스피" in blob:
        return "KOSPI"
    return "UNKNOWN"


def _kr_benchmark_used(segment: str) -> str:
    return {"KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ"}.get(segment, "UNKNOWN")


# ─── 3. per-row batch enrichment ─────────────────────────────────────


def _extract_ticker(row, market: str) -> str:
    tk = _row_get(row, "ticker")
    if tk is None:
        tk = getattr(row, "name", None)  # pandas Series index fallback
    tk = str(tk).strip() if tk is not None else ""
    if market == "KR":
        return tk.zfill(6) if tk.isdigit() else tk
    return tk.upper()


def _safe_call(fn, ticker: str) -> dict:
    try:
        out = fn(ticker)
        return out if isinstance(out, dict) else {"event_score": None, "error": None}
    except Exception as exc:
        return {"event_score": None, "error": f"{ticker}: fetch 실패 — {exc}"}


def _lazy_edgar():
    from screener.events_us import fetch_us_edgar_events
    return fetch_us_edgar_events


def _lazy_earnings():
    from screener.events_us import fetch_us_earnings_catalyst
    return fetch_us_earnings_catalyst


def _lazy_dart():
    from screener.events import fetch_kr_dart_events
    return fetch_kr_dart_events


def _derive_status(ctx: dict) -> str:
    if ctx.get("event_score") is not None:
        return "ok"
    flat = _flatten_error(ctx.get("error"))
    if flat:
        if "API 키" in flat or "DART_API_KEY" in flat:
            return "missing"
        if "CIK" in flat or "찾을 수 없" in flat or "데이터 없음" in flat:
            return "missing"
        return "error"
    return "unavailable"


def build_shadow_event_row(row, market: str, fetchers=None, now=None) -> dict:
    """
    Enrich a single candidate row with shadow event context.

    `fetchers` (dependency injection): optional dict with keys
      "edgar", "earnings" (US) / "dart" (KR). When absent, real fetchers are
      lazily imported. Tests pass fakes so nothing hits the network.

    Fail-soft: a fetcher exception yields status "error" (never raises).
    """
    now = now or datetime.now()
    market = (market or "US").upper()
    ticker = _extract_ticker(row, market)
    name   = _row_get(row, "name") or _row_get(row, "company") or ticker
    validated = _to_float_or_none(_row_get(row, "validated_score"))
    base_for_adj = validated if validated is not None else 0.0

    out: dict = {
        "ticker":                      ticker,
        "name":                        str(name),
        "market":                      market,
        "validated_score":             validated,
        "shadow_event_score":          None,
        "shadow_edgar_score":          None,
        "shadow_earnings_score":       None,
        "shadow_dart_score":           None,
        "shadow_event_adjusted_score": round(base_for_adj, 1) if validated is not None else None,
        "shadow_event_delta":          0.0,
        "shadow_event_reason":         "",
        "shadow_event_flags":          [],
        "shadow_event_status":         "unavailable",
        "shadow_event_error":          None,
        "shadow_event_fetched_at":     now.strftime("%Y-%m-%d %H:%M"),
    }

    # KR market-segment tag is set up-front so the columns always exist.
    if market == "KR":
        seg = infer_kr_market_segment(row)
        out["kr_market_segment"] = seg
        out["benchmark_used"]    = _kr_benchmark_used(seg)

    fetchers = fetchers or {}
    try:
        if market == "US":
            edgar_fn = fetchers.get("edgar") or _lazy_edgar()
            earn_fn  = fetchers.get("earnings") or _lazy_earnings()
            edgar_ctx = _safe_call(edgar_fn, ticker)
            earn_ctx  = _safe_call(earn_fn, ticker)
            event_ctx = combine_us_event_context(edgar_ctx, earn_ctx)
            out["shadow_edgar_score"]    = event_ctx.get("edgar_score")
            out["shadow_earnings_score"] = event_ctx.get("earnings_score")
        else:  # KR
            dart_fn  = fetchers.get("dart") or _lazy_dart()
            dart_ctx = _safe_call(dart_fn, ticker)
            event_ctx = normalize_kr_event_context(dart_ctx)
            out["shadow_dart_score"] = event_ctx.get("dart_score")

        out["shadow_event_score"]  = event_ctx.get("event_score")
        out["shadow_event_flags"]  = list(event_ctx.get("event_flags") or [])
        out["shadow_event_status"] = _derive_status(event_ctx)
        out["shadow_event_error"]  = _flatten_error(event_ctx.get("error"))

        adj = calc_event_adjusted_shadow_score(base_for_adj, event_ctx, row=row)
        out["shadow_event_adjusted_score"] = adj["event_adjusted_score"]
        out["shadow_event_delta"]          = adj["event_delta"]
        out["shadow_event_reason"]         = adj["event_adjustment_reason"]
    except Exception as exc:  # defensive — should not happen given _safe_call
        out["shadow_event_status"] = "error"
        out["shadow_event_error"]  = str(exc)

    return out


# ─── 4. local cache ──────────────────────────────────────────────────


def _date_str(d=None) -> str:
    if d is None:
        return datetime.now().strftime("%Y-%m-%d")
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def get_shadow_cache_key(market: str, ticker: str, date=None) -> str:
    """Cache key: '{MARKET}:{ticker}:{YYYY-MM-DD}'."""
    return f"{str(market).upper()}:{ticker}:{_date_str(date)}"


def load_shadow_event_cache(path=None) -> dict:
    """Load cache dict. Corrupt / missing file → empty dict (rebuild)."""
    p = Path(path) if path else DEFAULT_CACHE_PATH
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("shadow cache 읽기 실패 (무시): %s", exc)
    return {}


def save_shadow_event_cache(cache: dict, path=None) -> None:
    """Persist cache dict. Failures are swallowed (non-fatal)."""
    p = Path(path) if path else DEFAULT_CACHE_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.debug("shadow cache 저장 실패 (무시): %s", exc)


# ─── 5. batch enrichment over candidates ─────────────────────────────


def _iter_candidate_rows(df, limit: int):
    """Yield up to `limit` rows (pandas Series or dicts) in current order."""
    try:
        head = df.head(limit)
        return [r for _, r in head.iterrows()]
    except Exception:
        try:
            return list(df)[:limit]
        except Exception:
            return []


def enrich_shadow_events_for_candidates(
    df,
    market: str,
    limit: int = 50,
    force_refresh: bool = False,
    cache_path=None,
    fetchers=None,
    now=None,
) -> "pd.DataFrame":
    """
    Build a shadow-event DataFrame for the top `limit` candidates (current order).

    SHADOW-ONLY: does not mutate `df`, does not touch real ranking/selection.
    Adds original_rank, shadow_rank, shadow_rank_delta and shadow_candidate_action.
    Continues past per-ticker failures. Uses / updates the local cache unless
    force_refresh is set.
    """
    now = now or datetime.now()
    market = (market or "US").upper()

    if df is None:
        return pd.DataFrame()
    try:
        if len(df) == 0:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    cache    = load_shadow_event_cache(cache_path)
    date_str = _date_str(now)
    out_rows = []

    for original_rank, row in enumerate(_iter_candidate_rows(df, limit), 1):
        ticker = _extract_ticker(row, market)
        key    = get_shadow_cache_key(market, ticker, date_str)

        if (not force_refresh) and key in cache:
            shadow = dict(cache[key])
        else:
            shadow = build_shadow_event_row(row, market, fetchers=fetchers, now=now)
            cache[key] = shadow

        shadow = dict(shadow)
        shadow["original_rank"] = original_rank
        out_rows.append(shadow)

    save_shadow_event_cache(cache, cache_path)

    result = pd.DataFrame(out_rows)
    if result.empty:
        return result

    # shadow_rank: sort by adjusted shadow score desc; missing → validated_score.
    def _eff_score(r) -> float:
        s = r.get("shadow_event_adjusted_score")
        if s is None:
            s = r.get("validated_score")
        f = _to_float_or_none(s)
        return f if f is not None else float("-inf")

    result["_eff"] = result.apply(_eff_score, axis=1)
    result = result.sort_values("_eff", ascending=False, kind="stable").reset_index(drop=True)
    result["shadow_rank"]       = range(1, len(result) + 1)
    result["shadow_rank_delta"] = result["original_rank"] - result["shadow_rank"]

    def _action(r) -> str:
        delta      = _to_float_or_none(r.get("shadow_event_delta")) or 0.0
        rank_delta = int(r.get("shadow_rank_delta") or 0)
        if delta >= 5 or rank_delta >= 5:
            return "upgrade_watch"
        if delta <= -5 or rank_delta <= -5:
            return "downgrade_watch"
        return "unchanged"

    result["shadow_candidate_action"] = result.apply(_action, axis=1)
    result = result.drop(columns=["_eff"])

    # Final output preserves original candidate order.
    result = result.sort_values("original_rank", kind="stable").reset_index(drop=True)
    return result
