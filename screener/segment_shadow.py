"""
screener/segment_shadow.py
KR Segment Shadow Composite v1 — shadow-only, market-segment-specific composite.

Uses the existing `kr_market_segment` tag (KOSPI / KOSDAQ / UNKNOWN) to bias a
shadow composite score with transparent, inspectable heuristic rules:

  KOSPI  → "kospi_trend_supply_v1": trend/leader/supply persistence matter more.
  KOSDAQ → "kosdaq_breakout_event_risk_v1": breakout/volume/event-risk matter more.
  UNKNOWN→ "kr_generic_event_shadow_v1": fall back to the generic event-adjusted score.

CONTRACT
--------
SHADOW-ONLY / measurement-only. Nothing here mutates validated_score / final_score /
rr_score / catalyst_score, changes real ranking / candidate selection, or touches the
backtest formulas. No network calls, no external data, no Streamlit dependency.
These are v1 heuristics for observable A/B comparison — NOT production ranking.
"""
import logging
import math

from screener.event_shadow import infer_kr_market_segment
from screener.adverse_events import classify_kr_adverse_event_risk

logger = logging.getLogger(__name__)

KOSPI_PROFILE   = "kospi_trend_supply_v1"
KOSDAQ_PROFILE  = "kosdaq_breakout_event_risk_v1"
GENERIC_PROFILE = "kr_generic_event_shadow_v1"

# Financing / dilution / ownership-risk keywords scanned in shadow event flags.
_RISK_FLAG_KW_KR = ("유상증자", "전환사채", "신주인수권", "최대주주변경", "처분", "감소")
_RISK_FLAG_KW_EN = ("BW", "CB")

# Neutral defaults so missing factor columns never crash and never falsely trigger.
_FACTOR_DEFAULTS = {
    "leader_score":   50.0,
    "momentum_score": 50.0,
    "supply_score":   0.0,
    "supply_flow":    0.0,
    "sector_score":   0.0,
    "sect_rs":        0.0,
    "top_risk_score": 30.0,
    "volume_score":   50.0,
    "breakout_score": 0.0,
    "volume_shock":   0.0,
    "near_high":      0.0,
    "brk_persist":    0.0,
    "momentum_accel": 0.0,
    "buyable_score":  50.0,
}


# ─── small helpers (kept local to avoid coupling to private API) ─────


def _num(x, nd=None):
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
    if row is None:
        return default
    try:
        v = row.get(key, default) if hasattr(row, "get") else row[key]
    except Exception:
        return default
    return default if v is None else v


def _factor(row, key):
    """Read a factor with a neutral default; never raises."""
    v = _num(_rget(row, key))
    return v if v is not None else _FACTOR_DEFAULTS.get(key, 0.0)


def _norm_kr_ticker(value) -> str:
    t = "" if value is None else str(value).strip()
    return t.zfill(6) if t.isdigit() else t


def _flags_text(event_row) -> "tuple[str, str]":
    flags = _rget(event_row, "shadow_event_flags", []) if event_row is not None else []
    if isinstance(flags, str):
        flags = [flags]
    try:
        text = " ".join(str(f) for f in flags)
    except Exception:
        text = ""
    return text, text.upper()


def _has_financing_risk(event_row) -> bool:
    text, upper = _flags_text(event_row)
    if any(kw in text for kw in _RISK_FLAG_KW_KR):
        return True
    return any(kw in upper for kw in _RISK_FLAG_KW_EN)


# ─── 1. segment composite ────────────────────────────────────────────


def _resolve_segment(row, event_row) -> str:
    seg = _rget(event_row, "kr_market_segment") if event_row is not None else None
    if seg and str(seg).upper() in ("KOSPI", "KOSDAQ", "UNKNOWN"):
        return str(seg).upper()
    return infer_kr_market_segment(row)


def _base_scores(row, event_row) -> "tuple[float, float | None]":
    """Return (base_for_composite, validated_score). base prefers event-adjusted."""
    val = _num(_rget(row, "validated_score"))
    if val is None and event_row is not None:
        val = _num(_rget(event_row, "validated_score"))
    ev_adj = _num(_rget(event_row, "shadow_event_adjusted_score")) if event_row is not None else None
    base = ev_adj if ev_adj is not None else (val if val is not None else 50.0)
    return base, val


def _adverse_extra_penalty(adverse, levels, exclude_categories, cap) -> float:
    """
    Shadow-only adverse-event penalty from the v2 classifier, capped so it does
    not aggressively double-count the existing event_delta / financing components.

    Fires only when the classifier level is in `levels` and at least one matched
    category is NOT in `exclude_categories`. `cap` is the most-negative bound.
    """
    if not adverse:
        return 0.0
    level = adverse.get("adverse_event_risk_level")
    if level not in levels:
        return 0.0
    cats = set(adverse.get("adverse_event_categories", []))
    if not (cats - set(exclude_categories)):
        return 0.0
    pen = float(adverse.get("adverse_event_penalty_hint", 0) or 0)
    return max(float(cap), pen)  # cap is a floor on magnitude


def _kospi_components(row, event_row, adverse=None) -> list:
    comps = []
    leader = _factor(row, "leader_score")
    mom    = _factor(row, "momentum_score")
    supply = max(_factor(row, "supply_score"), _factor(row, "supply_flow"))
    sector = max(_factor(row, "sector_score"), _factor(row, "sect_rs"))
    risk   = _factor(row, "top_risk_score")
    vol    = _factor(row, "volume_score")
    ev_delta = _num(_rget(event_row, "shadow_event_delta")) if event_row is not None else None

    if leader >= 75:
        comps.append(("리더/추세 강세", 4.0, f"leader_score {leader:.0f}≥75"))
    if mom >= 65:
        comps.append(("모멘텀 지속", 3.0, f"momentum_score {mom:.0f}≥65"))
    if supply >= 65:
        comps.append(("수급 지속성", 4.0, f"supply {supply:.0f}≥65"))
    if sector >= 60:
        comps.append(("섹터 상대강도", 2.0, f"sector {sector:.0f}≥60"))
    if risk >= 85 and leader < 70:
        comps.append(("과열+약한추세", -4.0, f"risk {risk:.0f}≥85 & leader {leader:.0f}<70"))
    if ev_delta is not None and ev_delta <= -6:
        comps.append(("이벤트 부정", -5.0, f"shadow_event_delta {ev_delta:.0f}≤-6"))
    if vol < 45:
        comps.append(("거래량 약화", -3.0, f"volume_score {vol:.0f}<45"))
    # KOSPI: apply only medium/high/critical adverse events; ignore low-confidence
    # watch-only signals. Gentler cap (trend-tolerant profile).
    adv = _adverse_extra_penalty(
        adverse, levels={"medium", "high", "critical"},
        exclude_categories=set(), cap=-5.0)
    if adv < 0:
        comps.append(("악재 이벤트(분류기)", adv,
                      f"{adverse.get('adverse_event_risk_level')} "
                      f"{adverse.get('adverse_event_categories')}"))
    return comps


def _kosdaq_components(row, event_row, adverse=None) -> list:
    comps = []
    brk    = _factor(row, "breakout_score")
    vol    = _factor(row, "volume_score")
    vshock = _factor(row, "volume_shock")
    nearhi = _factor(row, "near_high")
    persist = _factor(row, "brk_persist")
    accel  = _factor(row, "momentum_accel")
    risk   = _factor(row, "top_risk_score")
    buyable = _factor(row, "buyable_score")
    ev_delta = _num(_rget(event_row, "shadow_event_delta")) if event_row is not None else None

    if brk >= 65:
        comps.append(("돌파 강세", 5.0, f"breakout_score {brk:.0f}≥65"))
    if vol >= 70 or vshock >= 70:
        comps.append(("거래량 확인", 4.0, f"volume {max(vol, vshock):.0f}≥70"))
    if nearhi >= 65 or persist >= 65:
        comps.append(("신고가/돌파지속", 3.0, f"near/persist {max(nearhi, persist):.0f}≥65"))
    if accel >= 60:
        comps.append(("모멘텀 가속", 2.0, f"momentum_accel {accel:.0f}≥60"))
    if risk >= 80:
        comps.append(("고위험", -6.0, f"top_risk_score {risk:.0f}≥80"))
    if ev_delta is not None and ev_delta <= -6:
        comps.append(("이벤트 부정", -8.0, f"shadow_event_delta {ev_delta:.0f}≤-6"))
    if _has_financing_risk(event_row):
        comps.append(("증자/희석/오너십 리스크", -5.0, "financing/dilution flag"))
    if buyable < 45:
        comps.append(("매수적합도 약화", -4.0, f"buyable_score {buyable:.0f}<45"))
    # KOSDAQ: extra penalty for SEVERE (high/critical) non-financing adverse events
    # (listing/regulatory/business/ownership). Dilution is excluded to avoid
    # double-counting the financing flag above. Capped to -6.
    adv = _adverse_extra_penalty(
        adverse, levels={"high", "critical"},
        exclude_categories={"dilution_financing"}, cap=-6.0)
    if adv < 0:
        comps.append(("심각 악재 이벤트(분류기)", adv,
                      f"{adverse.get('adverse_event_risk_level')} "
                      f"{adverse.get('adverse_event_categories')}"))
    return comps


def calc_kr_segment_shadow_score(row, event_row=None) -> dict:
    """
    Compute a shadow-only, segment-specific composite for one KR candidate.
    Does not mutate inputs. Clamps 0–100.
    """
    segment = _resolve_segment(row, event_row)
    base, val = _base_scores(row, event_row)

    # Adverse-event classification (shadow-only) from the event row's flags.
    adverse = classify_kr_adverse_event_risk(
        event_flags=_rget(event_row, "shadow_event_flags") if event_row is not None else None)

    out = {
        "kr_segment":                 segment,
        "kr_segment_shadow_score":    None,
        "kr_segment_shadow_delta":    0.0,
        "kr_segment_shadow_reason":   "",
        "kr_segment_shadow_profile":  GENERIC_PROFILE,
        "kr_segment_shadow_components": [],
        # adverse-event fields (all segments; JSON-serializable)
        "adverse_event_risk_level":   adverse["adverse_event_risk_level"],
        "adverse_event_risk_score":   adverse["adverse_event_risk_score"],
        "adverse_event_categories":   adverse["adverse_event_categories"],
        "adverse_event_penalty_hint": adverse["adverse_event_penalty_hint"],
    }

    if segment == "KOSPI":
        out["kr_segment_shadow_profile"] = KOSPI_PROFILE
        comps = _kospi_components(row, event_row, adverse)
    elif segment == "KOSDAQ":
        out["kr_segment_shadow_profile"] = KOSDAQ_PROFILE
        comps = _kosdaq_components(row, event_row, adverse)
    else:  # UNKNOWN → generic event-adjusted base; adverse fields included, score unchanged
        score = max(0.0, min(100.0, base))
        out["kr_segment_shadow_score"]  = round(score, 1)
        out["kr_segment_shadow_delta"]  = 0.0
        out["kr_segment_shadow_reason"] = "세그먼트 UNKNOWN — generic event-adjusted shadow 사용"
        return out

    total = sum(c[1] for c in comps)
    score = max(0.0, min(100.0, base + total))
    components = [{"label": c[0], "delta": round(float(c[1]), 1), "reason": c[2]} for c in comps]

    if components:
        reason = f"{segment} {out['kr_segment_shadow_profile']}: " + " / ".join(
            f"{c['label']}{c['delta']:+.0f}" for c in components
        )
    else:
        reason = f"{segment} {out['kr_segment_shadow_profile']}: 트리거 없음 (base 유지)"

    out["kr_segment_shadow_score"]      = round(score, 1)
    out["kr_segment_shadow_delta"]      = round(score - base, 1)
    out["kr_segment_shadow_reason"]     = reason
    out["kr_segment_shadow_components"] = components
    return out


# ─── 3. batch application ────────────────────────────────────────────


def _build_event_map(shadow_df) -> dict:
    """ticker(normalized) → event_row dict from a shadow batch df."""
    out: dict = {}
    if shadow_df is None:
        return out
    try:
        rows = [r for _, r in shadow_df.iterrows()]
    except Exception:
        try:
            rows = list(shadow_df)
        except Exception:
            rows = []
    for r in rows:
        tk = _norm_kr_ticker(_rget(r, "ticker"))
        if tk and tk not in out:  # first occurrence wins (duplicate-safe)
            out[tk] = r
    return out


def apply_kr_segment_shadow_composite(validated_df, shadow_df=None, limit=None):
    """
    Apply calc_kr_segment_shadow_score to KR candidates → shadow comparison df.
    Does not mutate inputs. Preserves original order with original_rank.
    """
    import pandas as pd

    if validated_df is None:
        return pd.DataFrame()
    try:
        records = [r for _, r in validated_df.iterrows()]
    except Exception:
        try:
            records = list(validated_df)
        except Exception:
            records = []
    if not records:
        return pd.DataFrame()

    if limit is not None:
        records = records[:limit]

    event_map = _build_event_map(shadow_df)

    out_rows = []
    for rank, row in enumerate(records, 1):
        tk = _norm_kr_ticker(_rget(row, "ticker"))
        event_row = event_map.get(tk)
        seg = calc_kr_segment_shadow_score(row, event_row)

        ev_adj  = _num(_rget(event_row, "shadow_event_adjusted_score")) if event_row is not None else None
        ev_delta = _num(_rget(event_row, "shadow_event_delta")) if event_row is not None else None
        flags = _rget(event_row, "shadow_event_flags", []) if event_row is not None else []
        if isinstance(flags, str):
            flags = [flags]
        flags = [str(f) for f in (flags or [])]

        out_rows.append({
            "original_rank":               rank,
            "ticker":                      tk,
            "name":                        str(_rget(row, "name", tk)),
            "kr_segment":                  seg["kr_segment"],
            "validated_score":             _num(_rget(row, "validated_score")),
            "shadow_event_adjusted_score": ev_adj,
            "shadow_event_delta":          ev_delta,
            "kr_segment_shadow_score":     seg["kr_segment_shadow_score"],
            "kr_segment_shadow_delta":     seg["kr_segment_shadow_delta"],
            "kr_segment_shadow_profile":   seg["kr_segment_shadow_profile"],
            "kr_segment_shadow_reason":    seg["kr_segment_shadow_reason"],
            "kr_segment_shadow_components": seg["kr_segment_shadow_components"],
            "shadow_event_flags":          flags,
            "adverse_event_risk_level":    seg.get("adverse_event_risk_level"),
            "adverse_event_risk_score":    seg.get("adverse_event_risk_score"),
            "adverse_event_categories":    seg.get("adverse_event_categories"),
            "adverse_event_penalty_hint":  seg.get("adverse_event_penalty_hint"),
        })

    result = pd.DataFrame(out_rows)
    if result.empty:
        return result

    # segment_shadow_rank: sort by composite desc (missing → validated → -inf)
    def _eff(r):
        s = r.get("kr_segment_shadow_score")
        if s is None:
            s = r.get("validated_score")
        f = _num(s)
        return f if f is not None else float("-inf")

    result["_eff"] = result.apply(_eff, axis=1)
    result = result.sort_values("_eff", ascending=False, kind="stable").reset_index(drop=True)
    result["segment_shadow_rank"]       = range(1, len(result) + 1)
    result["segment_shadow_rank_delta"] = result["original_rank"] - result["segment_shadow_rank"]

    def _action(r):
        delta = _num(r.get("kr_segment_shadow_delta")) or 0.0
        rank_delta = int(r.get("segment_shadow_rank_delta") or 0)
        if delta >= 5 or rank_delta >= 5:
            return "upgrade_watch"
        if delta <= -5 or rank_delta <= -5:
            return "downgrade_watch"
        return "unchanged"

    result["kr_segment_candidate_action"] = result.apply(_action, axis=1)
    result = result.drop(columns=["_eff"])
    result = result.sort_values("original_rank", kind="stable").reset_index(drop=True)

    cols = [
        "original_rank", "segment_shadow_rank", "segment_shadow_rank_delta",
        "ticker", "name", "kr_segment", "validated_score",
        "shadow_event_adjusted_score", "shadow_event_delta",
        "kr_segment_shadow_score", "kr_segment_shadow_delta",
        "kr_segment_shadow_profile", "kr_segment_shadow_reason",
        "kr_segment_shadow_components", "shadow_event_flags",
        "adverse_event_risk_level", "adverse_event_risk_score",
        "adverse_event_categories", "adverse_event_penalty_hint",
        "kr_segment_candidate_action",
    ]
    return result[[c for c in cols if c in result.columns]]
