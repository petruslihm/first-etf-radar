"""
screener/shadow_outcomes.py
Shadow A/B Outcome Tracker — forward-testing framework (NOT a historical backtest).

WHAT THIS IS
------------
After Top-N shadow event enrichment runs (see screener.event_shadow), this module
records a timestamped snapshot of:
  - the original validated Top-N ranking,
  - the event-adjusted shadow Top-N ranking,
  - overlap / additions / removals between the two,
  - KR market-segment tags,
  - an entry reference price per ticker,
and — *later*, once price data for the forward window exists — evaluates realized
5/10/20 trading-day forward returns for each group.

WHAT THIS IS NOT
----------------
This is forward outcome tracking, not a statistically valid historical backtest.
DART/EDGAR/Earnings signals are *current-state* signals; we do not pretend to
replay them historically. We record today's shadow selection and grade it later.

CONTRACT
--------
SHADOW-ONLY / measurement-only. Nothing here mutates validated_score / final_score /
rr_score / catalyst_score, changes ranking / candidate selection, or touches the
existing backtest formulas in screener/backtest.py. No Streamlit dependency. No
network calls at import; live price access is dependency-injected and optional.

PRICE FETCHER INTERFACE (dependency injection)
----------------------------------------------
A `price_fetcher` is any callable:

    price_fetcher(ticker: str, market: str, start_date: str) -> pandas.Series | None

It must return an ascending date-indexed Series of close prices beginning on/around
`start_date` (index position 0 == entry/reference day), or None when unavailable.
The evaluator reads:
    entry_price = snapshot row's entry_price if present and > 0, else series.iloc[0]
    exit_price for horizon h = series.iloc[h]   (h trading days after entry)

A `benchmark_fetcher` has the same shape but takes a benchmark label instead of a
ticker:  benchmark_fetcher(label: str, market: str, start_date: str) -> Series | None
"""
import json
import logging
import math
import statistics
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo-local cache (gitignored), consistent with screener/event_shadow.py.
_CACHE_DIR = Path(__file__).parent.parent / ".cache"
DEFAULT_SNAPSHOT_PATH = _CACHE_DIR / "shadow_outcome_snapshots.json"
DEFAULT_RESULT_PATH   = _CACHE_DIR / "shadow_outcome_results.json"

STORE_VERSION = 1

# Common price column names we will probe, in priority order.
_PRICE_COLS = ("price", "entry_price", "current_price", "close", "Close", "last", "현재가")


# ─── small utilities ─────────────────────────────────────────────────


def _num(x, nd=None):
    """Coerce to a plain python float (or None). NaN → None. Optional rounding."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd) if nd is not None else f


def _rget(row, key, default=None):
    """Defensive read from a pandas Series / dict-like row."""
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


def _norm_ticker(value, market: str) -> str:
    t = "" if value is None else str(value).strip()
    if (market or "").upper() == "KR":
        return t.zfill(6) if t.isdigit() else t
    return t.upper()


def _iter_records(df_or_list):
    """Yield rows (Series or dicts) from a DataFrame or list, in order."""
    if df_or_list is None:
        return []
    if isinstance(df_or_list, list):
        return list(df_or_list)
    try:
        return [r for _, r in df_or_list.iterrows()]
    except Exception:
        try:
            return list(df_or_list)
        except Exception:
            return []


def _json_default(o):
    """json.dump default hook for numpy / odd scalar types."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
    except Exception:
        pass
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


def _flags_to_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        return [v] if v else []
    try:
        return [str(x) for x in v]
    except Exception:
        return []


# ─── price-column resolution ─────────────────────────────────────────


def _resolve_price_col(records: list, explicit: "str | None") -> "str | None":
    if explicit:
        return explicit
    if not records:
        return None
    sample = records[0]
    try:
        keys = set(sample.keys()) if hasattr(sample, "keys") else set(getattr(sample, "index", []))
    except Exception:
        keys = set()
    for c in _PRICE_COLS:
        if c in keys:
            return c
    return None


def _build_price_map(records: list, market: str, price_col: "str | None") -> dict:
    """ticker(normalized) → (price, source). price None when unavailable."""
    out: dict = {}
    if price_col is None:
        return out
    for r in records:
        tk = _norm_ticker(_rget(r, "ticker"), market)
        if not tk:
            continue
        p = _num(_rget(r, price_col))
        if tk not in out or (out[tk][0] is None and p is not None):
            out[tk] = (p, price_col if p is not None else None)
    return out


# ─── 2. snapshot model ───────────────────────────────────────────────


# Optional KR segment composite fields that may be attached to snapshot rows.
_SEGMENT_FIELDS = (
    "kr_segment_shadow_score", "kr_segment_shadow_delta",
    "kr_segment_shadow_profile", "kr_segment_candidate_action",
)


def _row_view(row, rank, market, price_map, *, is_shadow, seg_map=None) -> dict:
    """One normalized row for validated_top / shadow_top."""
    tk = _norm_ticker(_rget(row, "ticker"), market)
    p, psrc = price_map.get(tk, (None, None))
    entry_p = _num(_rget(row, "entry_price")) if _rget(row, "entry_price") is not None else p
    entry_src = "row:entry_price" if (_rget(row, "entry_price") is not None) else (
        f"validated:{psrc}" if psrc else None
    )
    view = {
        "rank":            int(rank),
        "ticker":          tk,
        "name":            str(_rget(row, "name", tk)),
        "market":          (market or "").upper(),
        "validated_score": _num(_rget(row, "validated_score")),
        "entry_price":     entry_p,
        "entry_price_source": entry_src,
    }
    # optional shadow / segment fields when present
    seg = _rget(row, "kr_market_segment")
    if seg is not None:
        view["kr_market_segment"] = str(seg)
    bench = _rget(row, "benchmark_used")
    if bench is not None:
        view["benchmark_used"] = str(bench)
    for k in ("shadow_event_adjusted_score", "shadow_event_delta"):
        val = _rget(row, k)
        if val is not None:
            view[k] = _num(val)
    flags = _rget(row, "shadow_event_flags")
    if flags is not None:
        view["shadow_event_flags"] = _flags_to_list(flags)
    action = _rget(row, "shadow_candidate_action")
    if action is not None:
        view["shadow_candidate_action"] = str(action)
    # optional KR segment composite fields — from the row itself or seg_map
    for k in _SEGMENT_FIELDS:
        val = _rget(row, k)
        if val is None and seg_map is not None:
            val = (seg_map.get(tk) or {}).get(k)
        if val is not None:
            view[k] = _num(val) if k.endswith(("score", "delta")) else str(val)
    return view


def _order_shadow_records(records: list) -> list:
    """Order shadow rows by shadow_rank asc; fall back to adjusted score desc."""
    has_rank = any(_rget(r, "shadow_rank") is not None for r in records)
    if has_rank:
        return sorted(records, key=lambda r: (_num(_rget(r, "shadow_rank")) or 1e9))
    return sorted(
        records,
        key=lambda r: (_num(_rget(r, "shadow_event_adjusted_score"))
                       if _num(_rget(r, "shadow_event_adjusted_score")) is not None
                       else (_num(_rget(r, "validated_score")) or float("-inf"))),
        reverse=True,
    )


def build_shadow_ab_snapshot(
    validated_df,
    shadow_df,
    market: str,
    asof=None,
    top_n: int = 20,
    price_col: "str | None" = None,
    segment_df=None,
) -> dict:
    """
    Build a JSON-serializable A/B snapshot from the current validated ranking and
    the shadow-enriched ranking. Does not mutate inputs.

    Optional `segment_df` (output of apply_kr_segment_shadow_composite): when given,
    snapshot rows are annotated with kr_segment_shadow_score / _delta / _profile /
    kr_segment_candidate_action by ticker. Entirely optional — absence changes nothing.
    """
    market = (market or "US").upper()
    asof_dt = asof or datetime.now()
    if isinstance(asof_dt, date) and not isinstance(asof_dt, datetime):
        asof_dt = datetime(asof_dt.year, asof_dt.month, asof_dt.day)
    asof_date = asof_dt.strftime("%Y-%m-%d")
    asof_ts   = asof_dt.strftime("%Y-%m-%dT%H:%M:%S")
    snapshot_id = f"{market}_{asof_dt.strftime('%Y%m%dT%H%M%S')}_top{top_n}"

    val_records = _iter_records(validated_df)
    shadow_records = _iter_records(shadow_df)

    # Optional segment composite map: ticker → {segment fields}.
    seg_map = None
    if segment_df is not None:
        seg_map = {}
        for r in _iter_records(segment_df):
            tk = _norm_ticker(_rget(r, "ticker"), market)
            if not tk:
                continue
            seg_map[tk] = {k: _rget(r, k) for k in _SEGMENT_FIELDS if _rget(r, k) is not None}

    # Price map: prefer validated_df, fall back to anything in shadow_df.
    pcol_val = _resolve_price_col(val_records, price_col)
    price_map = _build_price_map(val_records, market, pcol_val)
    pcol_sh = _resolve_price_col(shadow_records, price_col)
    if pcol_sh:
        for tk, pair in _build_price_map(shadow_records, market, pcol_sh).items():
            if tk not in price_map or price_map[tk][0] is None:
                price_map[tk] = pair

    # validated_top: preserve current order unless an explicit 'rank' column exists.
    if val_records and _rget(val_records[0], "rank") is not None:
        val_records = sorted(val_records, key=lambda r: (_num(_rget(r, "rank")) or 1e9))
    validated_top = [
        _row_view(r, i, market, price_map, is_shadow=False, seg_map=seg_map)
        for i, r in enumerate(val_records[:top_n], 1)
    ]

    # shadow_top: use shadow_rank order if present.
    shadow_ordered = _order_shadow_records(shadow_records)
    shadow_top = [
        _row_view(r, i, market, price_map, is_shadow=True, seg_map=seg_map)
        for i, r in enumerate(shadow_ordered[:top_n], 1)
    ]

    val_set = [r["ticker"] for r in validated_top]
    sh_set  = [r["ticker"] for r in shadow_top]
    val_lookup = set(val_set)
    sh_lookup  = set(sh_set)

    overlap   = [t for t in sh_set if t in val_lookup]
    added     = [t for t in sh_set if t not in val_lookup]
    removed   = [t for t in val_set if t not in sh_lookup]

    deltas = [r.get("shadow_event_delta") for r in shadow_top
              if r.get("shadow_event_delta") is not None]
    avg_delta = _num(statistics.fmean(deltas), 3) if deltas else None
    up_cnt   = sum(1 for r in shadow_top if r.get("shadow_candidate_action") == "upgrade_watch")
    down_cnt = sum(1 for r in shadow_top if r.get("shadow_candidate_action") == "downgrade_watch")

    by_segment = {}
    if market == "KR":
        for r in shadow_top:
            seg = r.get("kr_market_segment", "UNKNOWN")
            by_segment[seg] = by_segment.get(seg, 0) + 1

    n_val, n_sh = len(validated_top), len(shadow_top)
    summary = {
        "count_validated":      n_val,
        "count_shadow":         n_sh,
        "overlap_count":        len(overlap),
        "overlap_ratio":        _num(len(overlap) / n_sh, 4) if n_sh else 0.0,
        "added_count":          len(added),
        "removed_count":        len(removed),
        "avg_shadow_delta":     avg_delta,
        "upgrade_watch_count":  up_cnt,
        "downgrade_watch_count": down_cnt,
    }
    if market == "KR":
        summary["by_segment"] = by_segment

    return {
        "snapshot_id":      snapshot_id,
        "asof_date":        asof_date,
        "asof_ts":          asof_ts,
        "market":           market,
        "top_n":            int(top_n),
        "validated_top":    validated_top,
        "shadow_top":       shadow_top,
        "overlap":          overlap,
        "added_by_shadow":  added,
        "removed_by_shadow": removed,
        "summary":          summary,
    }


# ─── 3. snapshot persistence ─────────────────────────────────────────


def _empty_store() -> dict:
    return {"version": STORE_VERSION, "snapshots": []}


def _load_store(path, key: str) -> dict:
    p = Path(path)
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get(key), list):
                data.setdefault("version", STORE_VERSION)
                return data
    except Exception as exc:
        logger.debug("shadow store 읽기 실패 (무시): %s", exc)
    return {"version": STORE_VERSION, key: []}


def _save_store(store: dict, path) -> None:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(store, ensure_ascii=False, default=_json_default),
                     encoding="utf-8")
    except Exception as exc:
        logger.debug("shadow store 저장 실패 (무시): %s", exc)


def load_shadow_outcome_store(path=None) -> dict:
    return _load_store(path or DEFAULT_SNAPSHOT_PATH, "snapshots")


def save_shadow_outcome_store(store, path=None) -> None:
    if not isinstance(store, dict) or "snapshots" not in store:
        store = _empty_store()
    _save_store(store, path or DEFAULT_SNAPSHOT_PATH)


def append_shadow_ab_snapshot(snapshot, path=None) -> dict:
    """
    Append a snapshot. Deterministic dedup: if a snapshot with the same
    snapshot_id already exists, it is NOT added again.
    Returns {"status": "added"|"duplicate"|"error", "snapshot_id": ...}.
    """
    if not isinstance(snapshot, dict) or not snapshot.get("snapshot_id"):
        return {"status": "error", "error": "invalid snapshot"}
    sid = snapshot["snapshot_id"]
    store = load_shadow_outcome_store(path)
    existing = {s.get("snapshot_id") for s in store["snapshots"]}
    if sid in existing:
        return {"status": "duplicate", "snapshot_id": sid}
    store["snapshots"].append(snapshot)
    save_shadow_outcome_store(store, path)
    return {"status": "added", "snapshot_id": sid}


def list_shadow_ab_snapshots(path=None, market=None, limit=None) -> list:
    store = load_shadow_outcome_store(path)
    snaps = store.get("snapshots", [])
    if market:
        m = market.upper()
        snaps = [s for s in snaps if str(s.get("market", "")).upper() == m]
    # newest first
    snaps = sorted(snaps, key=lambda s: str(s.get("asof_ts", "")), reverse=True)
    if limit is not None:
        snaps = snaps[:limit]
    return snaps


# ─── 3b. KR segment / action grouping & attribution helpers ──────────


# Snapshot-row fields that signal segment composite data is present.
_SEG_PRESENCE_FIELDS = (
    "kr_market_segment", "kr_segment_shadow_profile", "kr_segment_candidate_action",
)


def _has_segment_fields(rows) -> bool:
    return any(
        any(_rget(r, f) for f in _SEG_PRESENCE_FIELDS)
        for r in (rows or [])
    )


def group_snapshot_rows_by_segment_fields(rows) -> dict:
    """
    Group snapshot rows (e.g. snapshot["shadow_top"]) by KR segment composite
    fields. Pure: does not mutate rows. Missing fields fall back to
    "UNKNOWN" (segment) / "unclassified" (profile, action).

    Returns JSON-serializable lists of the original row references:
      {"by_segment": {...}, "by_profile": {...}, "by_action": {...},
       "by_segment_action": {seg: {action: [...]}}}
    """
    by_segment: dict = {}
    by_profile: dict = {}
    by_action:  dict = {}
    by_seg_act: dict = {}
    for r in (rows or []):
        seg  = str(_rget(r, "kr_market_segment") or "UNKNOWN")
        prof = str(_rget(r, "kr_segment_shadow_profile") or "unclassified")
        act  = str(_rget(r, "kr_segment_candidate_action") or "unclassified")
        by_segment.setdefault(seg, []).append(r)
        by_profile.setdefault(prof, []).append(r)
        by_action.setdefault(act, []).append(r)
        by_seg_act.setdefault(seg, {}).setdefault(act, []).append(r)
    return {
        "by_segment":        by_segment,
        "by_profile":        by_profile,
        "by_action":         by_action,
        "by_segment_action": by_seg_act,
    }


def _avg_field(rows_list, field):
    vals = [_num(_rget(r, field)) for r in rows_list]
    vals = [v for v in vals if v is not None]
    return _num(statistics.fmean(vals), 4) if vals else None


def _slim_segment_row(r) -> dict:
    return {
        "ticker":                      _rget(r, "ticker"),
        "name":                        _rget(r, "name"),
        "kr_market_segment":           _rget(r, "kr_market_segment"),
        "kr_segment_shadow_profile":   _rget(r, "kr_segment_shadow_profile"),
        "kr_segment_shadow_score":     _num(_rget(r, "kr_segment_shadow_score")),
        "kr_segment_shadow_delta":     _num(_rget(r, "kr_segment_shadow_delta")),
        "kr_segment_candidate_action": _rget(r, "kr_segment_candidate_action"),
    }


def summarize_segment_snapshot_attribution(snapshot) -> dict:
    """
    Count-level attribution of a snapshot's segment composite data, available
    BEFORE any forward price outcome exists. Uses snapshot["shadow_top"] rows
    (where the segment composite fields live). JSON-serializable; never mutates.
    """
    snapshot = snapshot or {}
    rows = snapshot.get("shadow_top", []) or []
    has_fields = _has_segment_fields(rows)

    if not has_fields:
        # No KR segment composite data — return an empty, clearly-flagged summary
        # rather than a misleading all-"UNKNOWN" bucket (e.g. for US snapshots).
        return {
            "snapshot_id":        snapshot.get("snapshot_id"),
            "market":             str(snapshot.get("market", "")).upper(),
            "count_rows":         len(rows),
            "has_segment_fields": False,
            "counts": {"by_segment": {}, "by_profile": {}, "by_action": {},
                       "by_segment_action": {}},
            "avg_kr_segment_shadow_delta": {"by_segment": {}, "by_action": {}},
            "avg_shadow_event_delta":      {"by_segment": {}, "by_action": {}},
            "top_upgrades": [],
            "top_downgrades": [],
        }

    groups = group_snapshot_rows_by_segment_fields(rows)

    def _counts(d):
        return {k: len(v) for k, v in d.items()}

    counts = {
        "by_segment": _counts(groups["by_segment"]),
        "by_profile": _counts(groups["by_profile"]),
        "by_action":  _counts(groups["by_action"]),
        "by_segment_action": {
            f"{seg}|{act}": len(rws)
            for seg, am in groups["by_segment_action"].items()
            for act, rws in am.items()
        },
    }

    avg_seg_delta = {
        "by_segment": {k: _avg_field(v, "kr_segment_shadow_delta")
                       for k, v in groups["by_segment"].items()},
        "by_action":  {k: _avg_field(v, "kr_segment_shadow_delta")
                       for k, v in groups["by_action"].items()},
    }
    avg_ev_delta = {
        "by_segment": {k: _avg_field(v, "shadow_event_delta")
                       for k, v in groups["by_segment"].items()},
        "by_action":  {k: _avg_field(v, "shadow_event_delta")
                       for k, v in groups["by_action"].items()},
    }

    ups   = [r for r in rows if _rget(r, "kr_segment_candidate_action") == "upgrade_watch"]
    downs = [r for r in rows if _rget(r, "kr_segment_candidate_action") == "downgrade_watch"]
    ups   = sorted(ups,   key=lambda r: _num(_rget(r, "kr_segment_shadow_delta")) or 0.0, reverse=True)
    downs = sorted(downs, key=lambda r: _num(_rget(r, "kr_segment_shadow_delta")) or 0.0)

    return {
        "snapshot_id":        snapshot.get("snapshot_id"),
        "market":             str(snapshot.get("market", "")).upper(),
        "count_rows":         len(rows),
        "has_segment_fields": _has_segment_fields(rows),
        "counts":             counts,
        "avg_kr_segment_shadow_delta": avg_seg_delta,
        "avg_shadow_event_delta":      avg_ev_delta,
        "top_upgrades":       [_slim_segment_row(r) for r in ups[:10]],
        "top_downgrades":     [_slim_segment_row(r) for r in downs[:10]],
    }


# ─── 4. outcome evaluation ───────────────────────────────────────────


def _benchmark_label_for(market: str, row_label: "str | None") -> str:
    if row_label and str(row_label).upper() != "UNKNOWN":
        return str(row_label).upper()
    return "KOSPI" if (market or "").upper() == "KR" else "SPY"


def _safe_series(fetcher, *args):
    if fetcher is None:
        return None
    try:
        return fetcher(*args)
    except Exception as exc:
        logger.debug("price fetcher 실패: %s", exc)
        return None


def _series_entry_exit(series, snapshot_entry, horizon):
    """Return (entry_price, exit_price_or_None) for a horizon from a close series."""
    if series is None:
        return snapshot_entry, None
    try:
        n = len(series)
    except Exception:
        return snapshot_entry, None
    if n == 0:
        return snapshot_entry, None
    try:
        first = _num(series.iloc[0])
    except Exception:
        first = None
    entry = snapshot_entry if (snapshot_entry is not None and snapshot_entry > 0) else first
    exit_p = None
    if n > horizon:
        try:
            exit_p = _num(series.iloc[horizon])
        except Exception:
            exit_p = None
    return entry, exit_p


def _aggregate_group(tickers, per_ticker, horizons, has_bench):
    out = {}
    for h in horizons:
        nets, benches, excesses = [], [], []
        n_missing = 0
        for t in tickers:
            pt = per_ticker.get(t)
            if not pt:
                n_missing += 1
                continue
            net = pt["ret"].get(h)
            if net is None:
                n_missing += 1
                continue
            nets.append(net)
            if has_bench:
                b = pt["bench"].get(h)
                if b is not None:
                    benches.append(b)
                    excesses.append(net - b)
        entry = {
            "avg_return":      _num(statistics.fmean(nets), 6) if nets else None,
            "median_return":   _num(statistics.median(nets), 6) if nets else None,
            "win_rate":        _num(sum(1 for x in nets if x > 0) / len(nets), 4) if nets else None,
            "hit_rate_5pct":   _num(sum(1 for x in nets if x >= 0.05) / len(nets), 4) if nets else None,
            "hit_rate_10pct":  _num(sum(1 for x in nets if x >= 0.10) / len(nets), 4) if nets else None,
            "worst_return":    _num(min(nets), 6) if nets else None,
            "count_available": len(nets),
            "count_missing":   n_missing,
        }
        if has_bench:
            entry["benchmark_return"] = _num(statistics.fmean(benches), 6) if benches else None
            entry["excess_return"]    = _num(statistics.fmean(excesses), 6) if excesses else None
        out[str(h)] = entry
    return out


def _aggregate_segment_rows(rows_list, per_ticker, horizons, has_bench) -> dict:
    """Per-horizon return stats for a group of snapshot rows, plus the group's
    average kr_segment_shadow_delta / shadow_event_delta (same across horizons)."""
    tickers = [_rget(r, "ticker") for r in rows_list]
    base = _aggregate_group(tickers, per_ticker, horizons, has_bench)
    avg_seg = _avg_field(rows_list, "kr_segment_shadow_delta")
    avg_ev  = _avg_field(rows_list, "shadow_event_delta")
    for hk in base:
        base[hk]["avg_kr_segment_shadow_delta"] = avg_seg
        base[hk]["avg_shadow_event_delta"]      = avg_ev
    return base


def _segment_attribution(sh_rows, per_ticker, horizons, has_bench) -> dict:
    """
    KR segment / profile / action / segment-action return attribution from
    snapshot["shadow_top"] rows. Returns {} when no segment fields are present.
    """
    if not _has_segment_fields(sh_rows):
        return {}
    groups = group_snapshot_rows_by_segment_fields(sh_rows)
    out = {
        "by_segment": {k: _aggregate_segment_rows(v, per_ticker, horizons, has_bench)
                       for k, v in groups["by_segment"].items()},
        "by_profile": {k: _aggregate_segment_rows(v, per_ticker, horizons, has_bench)
                       for k, v in groups["by_profile"].items()},
        "by_action":  {k: _aggregate_segment_rows(v, per_ticker, horizons, has_bench)
                       for k, v in groups["by_action"].items()},
        "by_segment_action": {},
    }
    for seg, am in groups["by_segment_action"].items():
        for act, rws in am.items():
            out["by_segment_action"][f"{seg}|{act}"] = _aggregate_segment_rows(
                rws, per_ticker, horizons, has_bench)
    return out


def evaluate_snapshot_outcomes(
    snapshot,
    price_fetcher,
    horizons=(5, 10, 20),
    benchmark_fetcher=None,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> dict:
    """
    Compute realized forward returns for validated_top vs shadow_top and related
    groups, using injected price fetchers. Returns in *fraction* units
    (0.05 == +5%). Not annualized; no Sharpe. Forward outcome tracking only.
    """
    snapshot = snapshot or {}
    market    = str(snapshot.get("market", "US")).upper()
    asof_date = snapshot.get("asof_date") or _today()
    horizons  = tuple(int(h) for h in horizons)
    cost_frac = (float(cost_bps or 0.0) + float(slippage_bps or 0.0)) / 10000.0
    has_bench = benchmark_fetcher is not None
    errors = []

    val_rows = snapshot.get("validated_top", []) or []
    sh_rows  = snapshot.get("shadow_top", []) or []

    # ticker → entry_price & benchmark label (prefer validated rows, then shadow)
    entry_map, bench_label = {}, {}
    for r in list(val_rows) + list(sh_rows):
        tk = r.get("ticker")
        if not tk:
            continue
        if tk not in entry_map or entry_map[tk] is None:
            entry_map[tk] = _num(r.get("entry_price"))
        if tk not in bench_label:
            bench_label[tk] = _benchmark_label_for(market, r.get("benchmark_used"))

    all_tickers = list(entry_map.keys())
    bench_series_cache: dict = {}

    per_ticker: dict = {}
    for tk in all_tickers:
        series = _safe_series(price_fetcher, tk, market, asof_date)
        snap_entry = entry_map.get(tk)
        rec = {"ret": {}, "bench": {}}
        if has_bench:
            label = bench_label.get(tk, _benchmark_label_for(market, None))
            if label not in bench_series_cache:
                bench_series_cache[label] = _safe_series(
                    benchmark_fetcher, label, market, asof_date)
            bser = bench_series_cache[label]
        else:
            bser = None
        for h in horizons:
            entry_p, exit_p = _series_entry_exit(series, snap_entry, h)
            if entry_p and entry_p > 0 and exit_p is not None:
                gross = exit_p / entry_p - 1.0
                rec["ret"][h] = round(gross - cost_frac, 6)
            else:
                rec["ret"][h] = None
            if has_bench and bser is not None:
                b_entry, b_exit = _series_entry_exit(bser, None, h)
                if b_entry and b_entry > 0 and b_exit is not None:
                    rec["bench"][h] = round(b_exit / b_entry - 1.0, 6)
                else:
                    rec["bench"][h] = None
            else:
                rec["bench"][h] = None
        per_ticker[tk] = rec

    val_t   = [r["ticker"] for r in val_rows]
    sh_t    = [r["ticker"] for r in sh_rows]
    overlap = snapshot.get("overlap", [])
    added   = snapshot.get("added_by_shadow", [])
    removed = snapshot.get("removed_by_shadow", [])

    results = {
        "validated_top": _aggregate_group(val_t, per_ticker, horizons, has_bench),
        "shadow_top":    _aggregate_group(sh_t, per_ticker, horizons, has_bench),
        "added_by_shadow":   _aggregate_group(added, per_ticker, horizons, has_bench),
        "removed_by_shadow": _aggregate_group(removed, per_ticker, horizons, has_bench),
        "overlap":           _aggregate_group(overlap, per_ticker, horizons, has_bench),
    }

    # shadow_minus_validated: per-horizon diff of average (and excess) returns.
    smv = {}
    for h in horizons:
        hk = str(h)
        v = results["validated_top"].get(hk, {})
        s = results["shadow_top"].get(hk, {})
        diff = {}
        if v.get("avg_return") is not None and s.get("avg_return") is not None:
            diff["avg_return_diff"] = round(s["avg_return"] - v["avg_return"], 6)
        if has_bench and v.get("excess_return") is not None and s.get("excess_return") is not None:
            diff["excess_return_diff"] = round(s["excess_return"] - v["excess_return"], 6)
        smv[hk] = diff
    results["shadow_minus_validated"] = smv

    # KR segment / profile / action attribution (empty {} when not applicable).
    results["segment_attribution"] = _segment_attribution(
        sh_rows, per_ticker, horizons, has_bench)

    return {
        "snapshot_id":  snapshot.get("snapshot_id"),
        "market":       market,
        "asof_date":    asof_date,
        "evaluated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "horizons":     list(horizons),
        "params": {
            "cost_bps":      float(cost_bps or 0.0),
            "slippage_bps":  float(slippage_bps or 0.0),
            "has_benchmark": has_bench,
        },
        "results":      results,
        "errors":       errors,
    }


# ─── 5. outcome result persistence ───────────────────────────────────


def _result_signature(result: dict) -> str:
    p = result.get("params", {}) or {}
    return (f"{result.get('snapshot_id')}|c{p.get('cost_bps')}|"
            f"s{p.get('slippage_bps')}|b{int(bool(p.get('has_benchmark')))}|"
            f"h{','.join(str(x) for x in result.get('horizons', []))}")


def load_shadow_outcome_results(path=None) -> dict:
    return _load_store(path or DEFAULT_RESULT_PATH, "results")


def append_shadow_outcome_result(result, path=None) -> dict:
    """
    Append (or replace) an outcome result. Deterministic dedup by
    snapshot_id + params signature: an existing match is replaced in place.
    Returns {"status": "added"|"replaced"|"error", ...}.
    """
    if not isinstance(result, dict) or not result.get("snapshot_id"):
        return {"status": "error", "error": "invalid result"}
    path = path or DEFAULT_RESULT_PATH
    store = _load_store(path, "results")
    sig = _result_signature(result)
    status = "added"
    for i, r in enumerate(store["results"]):
        if _result_signature(r) == sig:
            store["results"][i] = result
            status = "replaced"
            break
    else:
        store["results"].append(result)
    _save_store(store, path)
    return {"status": status, "snapshot_id": result["snapshot_id"], "signature": sig}


def list_shadow_outcome_results(path=None, snapshot_id=None, market=None, limit=None) -> list:
    store = _load_store(path or DEFAULT_RESULT_PATH, "results")
    res = store.get("results", [])
    if snapshot_id:
        res = [r for r in res if r.get("snapshot_id") == snapshot_id]
    if market:
        m = market.upper()
        res = [r for r in res if str(r.get("market", "")).upper() == m]
    res = sorted(res, key=lambda r: str(r.get("evaluated_at", "")), reverse=True)
    if limit is not None:
        res = res[:limit]
    return res


# ─── 6. optional default price fetcher (NOT used in tests) ───────────


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


def default_price_fetcher(ticker, market, start_date, max_days: int = 40):
    """
    Thin optional default fetcher. US via yfinance. KR is deferred (returns None)
    to avoid heavy scraping here. Never used by tests; live UI evaluation is gated
    on data availability. Returns an ascending close Series from start_date or None.
    """
    market = (market or "US").upper()
    if market != "US":
        return None
    try:
        import pandas as pd
        import yfinance as yf
        from datetime import timedelta
        start = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
        end = start + timedelta(days=int(max_days) * 2 + 10)  # calendar pad for trading days
        df = yf.Ticker(str(ticker)).history(
            start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
            auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()
    except Exception as exc:
        logger.debug("default_price_fetcher 실패 (%s): %s", ticker, exc)
        return None
