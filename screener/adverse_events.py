"""
screener/adverse_events.py
KR Adverse Event Risk Classifier v2 — shadow-only, granular, inspectable.

Classifies KR DART/event signals into adverse-risk categories and a severity
level/score, with a suggested shadow penalty hint. Used to improve KOSDAQ/KR
downside filtering in the segment shadow composite. Transparent keyword rules,
no external data, no network calls, no Streamlit dependency.

CONTRACT: shadow-only. Nothing here mutates validated/final/rr/catalyst scores,
ranking, selection, or backtest formulas. NOT production ranking.

Output of classify_kr_adverse_event_risk():
  adverse_event_risk_level   : "none" | "watch" | "medium" | "high" | "critical"
  adverse_event_risk_score   : int 0-100 (higher = worse)
  adverse_event_categories   : list[str]
  adverse_event_reasons      : list[str]
  adverse_event_penalty_hint : 0 | -2 | -5 | -8 | -12 (suggested shadow penalty)
  adverse_event_confidence   : "low" | "medium" | "high"
"""
import logging
import re

logger = logging.getLogger(__name__)

# ── category labels (stable, JSON-friendly) ──────────────────────────
_CAT_LABEL = {
    "dilution":  "dilution_financing",
    "ownership": "ownership_control",
    "listing":   "listing_accounting_regulatory",
    "business":  "business_stress",
}

# ── tier ordering / bands / hints ────────────────────────────────────
_TIER_RANK = {"none": 0, "watch": 1, "medium": 2, "high": 3, "critical": 4}
# level → (base, floor, ceiling)
_BANDS = {
    "none":     (0,  0,  19),
    "watch":    (28, 20, 44),
    "medium":   (55, 45, 69),
    "high":     (78, 70, 89),
    "critical": (93, 90, 100),
}
_PENALTY    = {"none": 0, "watch": -2, "medium": -5, "high": -8, "critical": -12}
_CONFIDENCE = {"none": "low", "watch": "low", "medium": "medium",
               "high": "high", "critical": "high"}

# ── Korean keyword rules: (keyword[space-stripped], category, tier) ───
_KW_RULES = [
    # dilution / financing
    ("유상증자",   "dilution", "medium"),
    ("전환사채",   "dilution", "medium"),
    ("신주인수권", "dilution", "medium"),
    ("교환사채",   "dilution", "medium"),
    ("전환가액",   "dilution", "high"),
    ("리픽싱",     "dilution", "high"),
    # ownership / control
    ("최대주주변경", "ownership", "high"),
    ("경영권변경",   "ownership", "high"),
    ("담보제공",     "ownership", "watch"),
    # listing / accounting / regulatory
    ("상장폐지",   "listing", "critical"),
    ("거래정지",   "listing", "critical"),
    ("의견거절",   "listing", "critical"),
    ("부적정",     "listing", "critical"),
    ("횡령",       "listing", "critical"),
    ("배임",       "listing", "critical"),
    ("관리종목",   "listing", "high"),
    ("투자위험",   "listing", "high"),
    ("한정",       "listing", "high"),
    ("불성실공시", "listing", "medium"),
    ("투자경고",   "listing", "medium"),
    ("투자주의",   "listing", "watch"),
    # business stress
    ("회생절차",     "business", "critical"),
    ("파산",         "business", "critical"),
    ("채무불이행",   "business", "high"),
    ("영업정지",     "business", "high"),
    ("주요계약해지", "business", "medium"),
    ("공급계약해지", "business", "medium"),
    ("계약해지",     "business", "medium"),
    ("소송",         "business", "medium"),
]

# ── English abbreviation rules (word-boundary on uppercased text) ────
_EN_RULES = [
    (r"\bCB\b", "dilution", "high", "CB"),
    (r"\bBW\b", "dilution", "high", "BW"),
    (r"\bEB\b", "dilution", "high", "EB"),
]

# direction keywords for ownership disposal/decrease
_OWNERSHIP_DIR_KW = ("처분", "매도", "감소")
_OWNERSHIP_CONTEXT = ("대량보유", "소유주식변동", "지분")


def _collect_text(event_flags, event_names, events) -> str:
    """Flatten flags/names/event-dicts into one searchable string. Never raises."""
    parts: list[str] = []

    def _add(x):
        if x is None:
            return
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, (list, tuple, set)):
            for e in x:
                _add(e)
        elif isinstance(x, dict):
            for k in ("report_nm", "report_name", "name", "title", "report", "nm", "flag"):
                v = x.get(k)
                if v:
                    parts.append(str(v))
        else:
            try:
                parts.append(str(x))
            except Exception:
                pass

    for src in (event_flags, event_names, events):
        try:
            _add(src)
        except Exception:
            pass
    return " ".join(parts)


def _result(level, score, categories, reasons, penalty, confidence) -> dict:
    return {
        "adverse_event_risk_level":   level,
        "adverse_event_risk_score":   int(score),
        "adverse_event_categories":   list(categories),
        "adverse_event_reasons":      list(reasons),
        "adverse_event_penalty_hint": int(penalty),
        "adverse_event_confidence":   confidence,
    }


def classify_kr_adverse_event_risk(event_flags=None, event_names=None, events=None) -> dict:
    """
    Classify KR adverse event risk from flags / report names / event dicts.
    Pure, defensive (never raises), JSON-serializable output.
    """
    text  = _collect_text(event_flags, event_names, events)
    norm  = text.replace(" ", "")
    upper = text.upper()

    matched: list[tuple] = []  # (category_label, tier, keyword)

    for kw, cat, tier in _KW_RULES:
        if kw in norm:
            matched.append((_CAT_LABEL[cat], tier, kw))

    for pattern, cat, tier, kw in _EN_RULES:
        if re.search(pattern, upper):
            matched.append((_CAT_LABEL[cat], tier, kw))

    # large-scale rights offering → escalate to high
    if "유상증자" in norm and ("대규모" in norm or "대량" in norm):
        matched.append((_CAT_LABEL["dilution"], "high", "대규모유상증자"))

    # ownership direction (처분/매도/감소): medium if clear stake context, else watch
    dir_hit = next((k for k in _OWNERSHIP_DIR_KW if k in norm), None)
    if dir_hit:
        if any(c in norm for c in _OWNERSHIP_CONTEXT):
            matched.append((_CAT_LABEL["ownership"], "medium", f"지분{dir_hit}"))
        else:
            matched.append((_CAT_LABEL["ownership"], "watch", dir_hit))

    # ambiguous audit-opinion mention → watch (unless already a hard audit signal)
    hard_audit = any(k in norm for k in ("의견거절", "부적정", "한정"))
    if "감사의견" in norm and not hard_audit:
        matched.append((_CAT_LABEL["listing"], "watch", "감사의견"))

    if not matched:
        return _result("none", 0, [], [], 0, "low")

    level = max((m[1] for m in matched), key=lambda t: _TIER_RANK[t])
    categories = sorted({m[0] for m in matched})

    reasons, seen = [], set()
    for _cat, tier, kw in matched:
        rs = f"{kw}({tier})"
        if rs not in seen:
            seen.add(rs)
            reasons.append(rs)

    base, floor, ceil = _BANDS[level]
    distinct = len(categories)
    total = len(matched)
    score = base + 4 * (distinct - 1) + 2 * (total - distinct)
    score = max(floor, min(ceil, score))
    if level == "critical":           # critical dominates
        score = max(90, min(100, score))

    return _result(level, score, categories, reasons,
                   _PENALTY[level], _CONFIDENCE[level])
