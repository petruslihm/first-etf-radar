"""
app.py  v8 — MU식 리레이팅 후보 의사결정기

탭 3개:
  🎯 리레이팅 후보    — 메인. 오늘 MU처럼 갈 후보 통합 랭킹
  🤖 GPT 분석        — 저평가/리레이팅/균형형 3모드 프롬프트
  ⚙️ 고급/진단       — 백테스트, 섹터ETF, 테마입력, 설정
"""

import logging
import math
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from screener.collector import (
    fetch_kr_benchmarks, fetch_kr_universe,
    fetch_us_universe, get_kr_tickers, get_us_tickers,
)
from screener.engine import (
    run_kr_pipeline, run_us_pipeline, clear_all_cache, clear_scores_cache,
    analyze_sector_etfs, calc_universe_theme_strength,
)
from screener.paths import get_data_date, data_age_days, DATA_DIR
from screener import journal
from screener import portfolio
from screener import followup_prompts
from screener.reporter import save_chart, save_csv
from screener.ai_insight import (
    generate_prompts, parse_ai_response,
    GRADE_COLOR, grade_badge_html, PROMPT_LABELS, TYPE_ICON,
    generate_theme_scan_prompt, parse_theme_response,
    get_all_theme_tickers, get_context_factors, get_avoid_list,
    calc_theme_score,
)
from screener.backtest import (
    run_walkforward_backtest, optimize_weights,
    run_kr_walkforward_backtest, optimize_kr_weights,
    load_kr_backtest, load_optimal_weights_kr,
    load_weights_history,
    analyze_factor_power, analyze_factor_quintiles,
    analyze_factor_hit15, normalize_by_date,
    compute_custom_objective, walk_forward_optimization, save_optimal_weights,
    estimate_return, load_backtest, load_optimal_weights,
)
from screener.filters import (
    is_kr_backtest_excluded_ticker,
    is_kr_benchmark_code,
    is_kr_candidate_etf_name,
    is_kr_precheck_excluded_security_name,
    is_us_backtest_excluded_ticker,
    is_us_live_etf_ticker,
)
try:
    from utils.scoring import calc_validated_score, DEFAULT_WEIGHTS as _SCORE_WEIGHTS
except ImportError:
    from screener.backtest import calc_validated_score
    _SCORE_WEIGHTS = {}
from screener.analyst import fetch_analyst_batch, get_earnings_calendar

logging.basicConfig(level=logging.WARNING)
TODAY      = date.today().isoformat()
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# 재무 추출 헬퍼 (top-level)
# ════════════════════════════════════════════════════════════════════

def _extract_fund_from_df(full_df) -> dict:
    if full_df is None or isinstance(full_df, tuple):
        return {}
    try:
        if full_df.empty: return {}
    except Exception:
        return {}
    fund_cols = ["fund_score","fund_grade","rev_growth","op_growth",
                 "eps_growth","rev_accel","trailing_pe","forward_pe",
                 "peg","op_margin","gross_margin"]
    result = {}
    for ticker in full_df.index:
        row = {}
        for col in fund_cols:
            if col in full_df.columns:
                v = full_df.loc[ticker, col]
                if v is not None: row[col] = v
        if row.get("fund_grade") not in (None, ""):
            result[ticker] = row
    return result


def _n(x):
    try:
        return len(x) if x is not None and not isinstance(x, tuple) and not x.empty else 0
    except Exception:
        return 0


def _latest_saved_result_file(market: str) -> Path | None:
    prefix = "us_leader" if market == "US" else "kr_leader"
    files = sorted(REPORT_DIR.glob(f"{prefix}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_saved_results(market: str) -> tuple[bool, str]:
    path = _latest_saved_result_file(market)
    if path is None:
        return False, f"{market} 저장 결과 파일이 없습니다. 먼저 스크리너를 실행해 결과를 저장하세요."

    try:
        df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    except Exception as e:
        return False, f"저장 결과를 불러오지 못했습니다: {e}"

    if df.empty:
        return False, f"저장 결과 파일이 비어 있습니다: {path.name}"

    prefix = market.lower()
    df.index = df.index.map(str)
    st.session_state.update({
        f"{prefix}_leader": df,
        f"{prefix}_buyable": None,
        f"{prefix}_breakout": None,
        f"{prefix}_turnaround": None,
        f"{prefix}_full_df": df.copy(),
        f"{prefix}_fund": _extract_fund_from_df(df),
        f"{prefix}_analyzed_count": len(df),
        f"{prefix}_loaded_result_file": str(path),
        "scr_date": path.stem.rsplit("_", 1)[-1],
    })
    if market == "US":
        st.session_state.setdefault("us_analyst", {})
    return True, f"{market} 최신 저장 결과 {len(df)}개를 불러왔습니다: {path.name}"


# ════════════════════════════════════════════════════════════════════
# 세션 초기화
# ════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "market": "US",
        "us_leader":None,"us_buyable":None,"us_breakout":None,"us_turnaround":None,
        "kr_leader":None,"kr_buyable":None,"kr_breakout":None,"kr_turnaround":None,
        "us_full_df":None,"kr_full_df":None,
        "us_fund":{},"kr_fund":{},
        "us_analyst":{},"backtest_result":None,"backtest_kr_result":None,"snapshot_results":[],
        "us_themes":[],"kr_themes":[],
        "us_theme_tickers":[],"kr_theme_tickers":[],
        "us_context_factors":{},"kr_context_factors":{},
        "us_avoid_tickers":[],"kr_avoid_tickers":[],
        "us_theme_scores":[],"kr_theme_scores":[],
        "sector_analysis":[],"universe_themes_us":[],"universe_themes_kr":[],
        "scr_date":None,"failed_us":[],"failed_kr":[],
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════
# Re-rating Score  (가치평가와 완전 분리)
# ════════════════════════════════════════════════════════════════════

def calc_rerating_score(row: dict, fund: dict = None, analyst: dict = None) -> float:
    """
    MU식 리레이팅 가능성 점수 (0~100).

    기준:
      추정치 상향 가능성    30%
      실적 성장 가속도      20%
      테마 희소성           20%
      상대강도              15%
      거래량/돌파           10%
      밸류에이션 부담       -5% (약하게만)

    핵심: 과열이라도 추정치·테마가 계속 오르면 높은 점수
    """
    score = 50.0
    fd = fund or {}
    an = analyst or {}

    # ── 추정치 상향 가능성 (0~30점) ───────────────────────────────
    eps_rev = an.get("eps_revision","neutral")
    eps_trend = float(an.get("eps_trend", 0) or 0)
    beat = int(an.get("beat_count", 0) or 0)
    analyst_score = float(an.get("analyst_score", 50) or 50)

    if eps_rev == "up":
        score += 15 + min(15, eps_trend * 0.5)
    elif eps_rev == "down":
        score -= 15

    if beat >= 3: score += 8
    elif beat >= 2: score += 4

    # ── 실적 성장 가속 (0~20점) ────────────────────────────────────
    rev_accel = fd.get("rev_accel", False)
    fund_score = float(fd.get("fund_score", 50) or 50)
    rev_growth = float(fd.get("rev_growth", 0) or 0)

    if rev_accel: score += 12
    if math.isfinite(rev_growth):
        if rev_growth >= 30:  score += 8
        elif rev_growth >= 20: score += 5
        elif rev_growth >= 10: score += 2
        elif rev_growth < 0:   score -= 5
    score += (fund_score - 50) / 50 * 8

    # ── 테마/섹터 희소성 (0~20점) ──────────────────────────────────
    is_theme = bool(row.get("is_theme_pick", False))
    cat = float(row.get("catalyst_score", 50) or 50)

    if is_theme: score += 10
    score += (cat - 50) / 50 * 10

    # ── 상대강도 (0~15점) ──────────────────────────────────────────
    rs_pct = float(row.get("rs_rank_pct", 50) or 50)
    ret20  = float(row.get("ret_20d", 0) or 0)
    score += (rs_pct - 50) / 50 * 8
    score += min(7, max(-5, ret20 * 0.1))

    # ── 거래량/돌파 (0~10점) ───────────────────────────────────────
    brk = float(row.get("breakout_score", 0) or 0)
    vol = float(row.get("volume_score", 50) or 50)
    score += brk * 0.05
    score += (vol - 50) / 50 * 5

    # ── 밸류에이션 부담 (약한 패널티 -5점만) ────────────────────────
    risk = float(row.get("top_risk_score", 30) or 30)
    ret5 = float(row.get("ret_5d", 0) or 0)
    flag = str(row.get("risk_flag", "") or "")
    # 진짜 과열 + 추세 훼손 시만 감점
    if risk >= 90 and ret5 < 0:
        score -= 15
    elif risk >= 90 and ("음봉" in flag or "윗꼬리" in flag):
        score -= 10
    elif risk >= 80:
        score -= 3  # 과열이지만 추세 유지면 최소 감점

    return round(max(0, min(100, score)), 1)


# ════════════════════════════════════════════════════════════════════
# 단일 티커 검색 (전체 유니버스에서 찾기)
# ════════════════════════════════════════════════════════════════════

def _find_ticker_candidate(ticker_query: str, market: str = "KR") -> dict | None:
    """
    티커/종목코드로 단일 종목을 찾아 후보 dict 형태로 반환.
    1) 먼저 4개 트랙 후보(build_candidates)에서 검색
    2) 없으면 전체 유니버스(kr_full_df / us_full_df)에서 검색
    찾지 못하면 None.
    """
    q = str(ticker_query).strip().upper()
    if not q:
        return None

    # 1) 트랙 후보에서 먼저 검색
    for c in build_candidates(market, "전체", exclude_turnaround=False):
        if str(c["ticker"]).upper() == q:
            return c
        row_obj = c.get("row", {})
        row = dict(row_obj) if hasattr(row_obj, "items") else {}
        for col in ("ticker", "symbol", "code", "name", "company"):
            v = row.get(col)
            if v is not None and str(v).strip().upper() == q:
                return c

    # 2) 전체 유니버스에서 검색
    full_key = "kr_full_df" if market == "KR" else "us_full_df"
    full_df  = st.session_state.get(full_key)
    if full_df is None or isinstance(full_df, tuple):
        return None
    try:
        if full_df.empty:
            return None
    except Exception:
        return None

    fund_map = st.session_state.get("kr_fund" if market == "KR" else "us_fund", {})
    analyst  = {} if market == "KR" else st.session_state.get("us_analyst", {})

    # 종목코드/티커 정확 매칭
    match_ticker = None
    for t in full_df.index:
        if str(t).upper() == q:
            match_ticker = t
            break
    if match_ticker is None:
        for col in ("ticker", "symbol", "code", "name", "company"):
            if col not in full_df.columns:
                continue
            for t in full_df.index:
                value = str(full_df.loc[t, col] or "").strip()
                if value and (value.upper() == q or (col in ("name", "company") and q in value.upper())):
                    match_ticker = t
                    break
            if match_ticker is not None:
                break
    if match_ticker is None:
        return None

    row      = full_df.loc[match_ticker]
    fd       = fund_map.get(match_ticker, {})
    an       = analyst.get(match_ticker, {})
    row_dict = dict(row)
    if an:
        row_dict.update(an)

    rr_score = calc_rerating_score(row_dict, fd, an)
    exp      = estimate_return(row_dict)
    upside   = float(an.get("upside_pct", float("nan")) or float("nan"))
    if not math.isfinite(upside):
        upside = float(exp.get("expected_mid", float("nan")) or float("nan"))

    return {
        "ticker":   match_ticker,
        "row":      row,
        "track":    str(row.get("track", "") or ""),
        "rr_score": rr_score,
        "exp":      exp,
        "fund":     fd,
        "analyst":  an,
        "upside":   upside,
    }


def _analyze_us_ticker_ondemand(ticker: str) -> "tuple[dict | None, str]":
    """
    미국 티커를 on-demand로 분석한다.
    스크리너 결과에 없는 티커도 yfinance에서 직접 수집해
    기존 calc_us_factors + validated_score 경로를 재사용한다.
    실패해도 기존 스크리너 결과에 영향 없음.

    Returns: (found_dict, error_msg)
    found_dict은 _render_ticker_deep_dive와 호환되는 형태.
    """
    from screener.collector import fetch_single_us_ticker_ondemand
    from screener.factors import calc_us_factors
    from screener.universe import normalize_sector, SECTOR_ETF

    t = str(ticker).strip().upper()
    item, bench_close, err = fetch_single_us_ticker_ondemand(t)
    if item is None:
        return None, err or f"{t}: 데이터를 가져오지 못했습니다."

    # 섹터 벤치마크
    sector_norm  = normalize_sector(item.get("sector", ""))
    sector_bench = SECTOR_ETF.get(sector_norm, "SPY")

    def _sret(sym: str, d: int) -> float:
        # `or` 연산자를 쓰면 pandas Series가 boolean context에 올라가 오류 발생.
        # 반드시 `is None` 체크로 폴백한다.
        s = bench_close.get(sym)
        if s is None:
            s = bench_close.get("SPY")
        if s is not None and len(s) > d:
            try:
                return float((s.iloc[-1] / s.iloc[-(d + 1)] - 1) * 100)
            except Exception:
                pass
        return 0.0

    sector_avg_ret20 = _sret(sector_bench, 20)
    sector_avg_ret60 = _sret(sector_bench, 60)

    # rs_rank_pct: 로드된 스크리너의 RS 분포와 비교 (없으면 0.5)
    rs_rank_pct = 0.5
    full_df = st.session_state.get("us_full_df")
    try:
        if full_df is not None and not isinstance(full_df, tuple) and not full_df.empty:
            if "rs_spy_20" in full_df.columns:
                close_s = item["ohlcv"]["Close"]
                r20_t   = _sret(t, 20) if t in bench_close else (
                    float((close_s.iloc[-1] / close_s.iloc[-21] - 1) * 100)
                    if len(close_s) >= 21 else 0.0
                )
                r20_spy = _sret("SPY", 20)
                rs_t    = r20_t - r20_spy
                univ_rs = full_df["rs_spy_20"].dropna()
                if len(univ_rs) > 5:
                    rs_rank_pct = float((univ_rs < rs_t).mean())
    except Exception:
        pass

    try:
        raw_result = calc_us_factors(
            ticker=t,
            item=item,
            bench_close=bench_close,
            sector_bench=sector_bench,
            sector_avg_ret20=sector_avg_ret20,
            sector_avg_ret60=sector_avg_ret60,
            rs_rank_pct=rs_rank_pct,
        )
    except Exception as exc:
        return None, f"팩터 계산 실패: {exc}"

    # DataFrame/Series로 돌아오면 dict로 정규화 (방어 레이어)
    row = _normalize_ondemand_row(raw_result)
    if row is None:
        return None, f"{t}: 팩터 계산 결과를 정규화할 수 없습니다."

    try:
        rr = calc_rerating_score(row, {}, {})
    except Exception:
        rr = 50.0

    # 스크리너 상위 후보 대비 비교
    screener_comparison = None
    screener_p75        = None
    screener_median     = None
    try:
        if full_df is not None and not isinstance(full_df, tuple) and not full_df.empty:
            if "validated_score" in full_df.columns:
                scores = full_df["validated_score"].dropna()
                if len(scores) > 5:
                    screener_p75    = float(scores.quantile(0.75))
                    screener_median = float(scores.median())
                    vs = float(_safe_row_value(row, "validated_score", 0.0))
                    if vs >= screener_p75:
                        screener_comparison = "strong"
                    elif vs >= screener_median:
                        screener_comparison = "average"
                    else:
                        screener_comparison = "weak"
    except Exception:
        pass

    try:
        exp = estimate_return(row)
    except Exception:
        exp = {}
    try:
        upside_raw = exp.get("expected_mid")
        upside = float(upside_raw) if upside_raw is not None else float("nan")
        if not math.isfinite(upside):
            upside = float("nan")
    except Exception:
        upside = float("nan")

    track_val = _safe_row_value(row, "track", "")

    # EDGAR 이벤트 촉매 (non-blocking; 실패해도 기존 결과 유지)
    us_event_context: dict = {}
    try:
        from screener.events_us import fetch_us_edgar_events
        us_event_context = fetch_us_edgar_events(t)
    except Exception as _exc:
        us_event_context = {
            "ticker": t, "cik": None, "events": [], "event_score": None,
            "event_flags": [], "positive_events": [], "negative_events": [],
            "neutral_events": [], "error": str(_exc), "source": "SEC EDGAR",
        }

    # 실적 촉매 (non-blocking; 실패해도 기존 결과 유지)
    us_earnings_context: dict = {}
    try:
        from screener.events_us import fetch_us_earnings_catalyst
        us_earnings_context = fetch_us_earnings_catalyst(t, item=item)
    except Exception as _exc:
        us_earnings_context = {
            "ticker": t, "earnings_score": None, "earnings_flags": [],
            "latest_earnings_date": None, "eps_surprise_pct": None,
            "revenue_surprise_pct": None, "error": str(_exc), "source": "yfinance",
        }

    # 이벤트 조정 shadow score (on-demand 전용; 기존 score 필드 불변)
    _us_ev_adj = _calc_event_adjusted_score(
        base_score=float(_safe_row_value(row, "validated_score") or 0.0),
        event_context=us_event_context,
        risk_score=_safe_row_value(row, "top_risk_score"),
        market="US",
    )

    return {
        "ticker":                    t,
        "row":                       row,
        "track":                     str(track_val) if track_val else "",
        "rr_score":                  rr,
        "exp":                       exp,
        "fund":                      {},
        "analyst":                   {},
        "upside":                    upside,
        "_ondemand":                 True,
        "_screener_comparison":      screener_comparison,
        "_screener_p75":             screener_p75,
        "_screener_median":          screener_median,
        "us_event_context":          us_event_context,
        "us_earnings_context":       us_earnings_context,
        "event_adjusted_score":      _us_ev_adj["event_adjusted_score"],
        "event_delta":               _us_ev_adj["event_delta"],
        "event_adjustment_reason":   _us_ev_adj["event_adjustment_reason"],
        "event_score_source":        "SEC EDGAR",
    }, ""


# ── on-demand 결과 정규화 헬퍼 ────────────────────────────────────────

def _normalize_ondemand_row(result) -> "dict | None":
    """
    calc_us_factors() 반환값을 스칼라 값만 담긴 dict로 정규화한다.
    - dict      → 그대로 반환
    - pd.Series → to_dict()
    - 1-row DataFrame → iloc[0].to_dict()
    - empty DataFrame / None → None
    pandas Series를 UI·채점 함수에 절대 넘기지 않기 위한 방어 레이어.
    """
    if result is None:
        return None
    if isinstance(result, dict):
        return result
    if isinstance(result, pd.Series):
        return result.to_dict()
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return None
        return result.iloc[0].to_dict()
    return None


def _safe_row_value(row, key: str, default=None):
    """
    행 dict / Series에서 스칼라를 안전하게 꺼낸다.
    - pandas Series 값이면 .iloc[0]으로 단일 스칼라 추출
    - None · NaN이면 default 반환
    - boolean context에 Series를 절대 올리지 않는다
    """
    try:
        val = row.get(key, default) if hasattr(row, "get") else row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if val is None:
        return default
    if isinstance(val, pd.Series):
        if val.empty:
            return default
        val = val.iloc[0]
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
        fv = float(val)
        if math.isnan(fv):
            return default
    except (TypeError, ValueError):
        pass
    return val


# ── 이벤트 조정 shadow score ─────────────────────────────────────────

def _calc_event_adjusted_score(
    base_score: float,
    event_context: "dict",
    risk_score: "float | None" = None,
    market: str = "US",
) -> dict:
    """
    이벤트 데이터로 base_score를 조정한 shadow score를 계산한다.
    validated_score / rr_score / catalyst_score 등 기존 점수를 변경하지 않는다.
    결과는 on-demand 결과 dict에 별도 필드로만 붙는다.

    실제 델타 로직은 screener.event_shadow.calc_event_adjusted_shadow_score로
    위임한다. 여기서는 "이벤트 점수 없음 → event_adjusted_score=None" 이라는
    기존 UI 계약(없으면 shadow 카드 숨김)을 유지하기 위한 얇은 래퍼만 둔다.
    """
    from screener.event_shadow import calc_event_adjusted_shadow_score

    ev = event_context or {}
    raw = ev.get("event_score")
    usable = raw is not None
    if usable:
        try:
            float(raw)
        except (TypeError, ValueError):
            usable = False

    if not usable:
        # 기존 동작 보존: 이벤트 점수 없으면 조정 없음(None) 반환.
        return {
            "event_adjusted_score":    None,
            "event_delta":             0.0,
            "event_adjustment_reason": "이벤트 데이터 없음",
            "event_flags":             [],
            "event_risk_flags":        [],
        }

    return calc_event_adjusted_shadow_score(
        base_score,
        event_context,
        row={"top_risk_score": risk_score},
    )


# ── KR on-demand 헬퍼 / 분석 함수 ─────────────────────────────────────

def _is_kr_query(query: str) -> bool:
    """6자리 숫자코드 또는 한글 포함 여부로 KR 종목 조회 여부 판단."""
    import re as _re
    q = str(query).strip()
    if _re.match(r"^\d{6}$", q):
        return True
    if _re.search(r"[가-힣]", q):
        return True
    return False


def _analyze_kr_ticker_ondemand(query: str) -> "tuple[dict | None, str]":
    """
    KR 단일 종목 on-demand 분석.
    fetch_single_kr_ticker_ondemand → calc_kr_factors → _normalize_ondemand_row 순으로 처리한다.
    기존 screener 결과·session_state를 절대 변경하지 않는다.
    """
    from screener.collector import fetch_single_kr_ticker_ondemand
    from screener.factors import calc_kr_factors

    q = str(query).strip()
    item, bench_close, err = fetch_single_kr_ticker_ondemand(q)
    if item is None:
        return None, err or f"'{q}': 데이터를 가져오지 못했습니다."

    code    = item["ticker"]
    mkt_key = "kospi" if item.get("market", "KOSPI") == "KOSPI" else "kosdaq"

    # ─ 로드된 KR 스크리너에서 sector_avg_ret20 추정 ─────────────
    sector_avg_ret20 = 0.0
    full_df = st.session_state.get("kr_full_df")
    try:
        if full_df is not None and not isinstance(full_df, tuple) and not full_df.empty:
            if "ret_20d" in full_df.columns:
                sector_avg_ret20 = float(full_df["ret_20d"].dropna().mean())
    except Exception:
        pass

    # ─ 로드된 KR 스크리너에서 rs_rank_pct 추정 ──────────────────
    rs_rank_pct = 0.5
    try:
        if full_df is not None and not isinstance(full_df, tuple) and not full_df.empty:
            if "rs_mkt_20" in full_df.columns:
                close_s   = item["ohlcv"]["Close"]
                bench_s   = bench_close.get(mkt_key) or bench_close.get("kospi")
                if bench_s is not None and len(bench_s) >= 21 and len(close_s) >= 21:
                    r20_t  = float((close_s.iloc[-1] / close_s.iloc[-21] - 1) * 100)
                    r20_bk = float((bench_s.iloc[-1] / bench_s.iloc[-21] - 1) * 100)
                    rs_t   = r20_t - r20_bk
                    univ_rs = full_df["rs_mkt_20"].dropna()
                    if len(univ_rs) > 5:
                        rs_rank_pct = float((univ_rs < rs_t).mean())
    except Exception:
        pass

    try:
        raw_result = calc_kr_factors(
            ticker=code,
            item=item,
            bench_close=bench_close,
            sector_avg_ret20=sector_avg_ret20,
            rs_rank_pct=rs_rank_pct,
        )
    except Exception as exc:
        return None, f"팩터 계산 실패: {exc}"

    row = _normalize_ondemand_row(raw_result)
    if row is None:
        return None, f"{code}: 팩터 계산 결과를 정규화할 수 없습니다."

    try:
        rr = calc_rerating_score(row, {}, {})
    except Exception:
        rr = 50.0

    # ─ 로드된 스크리너 분포와 비교 ───────────────────────────────
    screener_comparison = screener_p75 = screener_median = None
    try:
        if full_df is not None and not isinstance(full_df, tuple) and not full_df.empty:
            if "validated_score" in full_df.columns:
                scores = full_df["validated_score"].dropna()
                if len(scores) > 5:
                    screener_p75    = float(scores.quantile(0.75))
                    screener_median = float(scores.median())
                    vs = float(_safe_row_value(row, "validated_score", 0.0))
                    if vs >= screener_p75:
                        screener_comparison = "strong"
                    elif vs >= screener_median:
                        screener_comparison = "average"
                    else:
                        screener_comparison = "weak"
    except Exception:
        pass

    try:
        exp = estimate_return(row)
    except Exception:
        exp = {}
    try:
        upside_raw = exp.get("expected_mid")
        upside = float(upside_raw) if upside_raw is not None else float("nan")
        if not math.isfinite(upside):
            upside = float("nan")
    except Exception:
        upside = float("nan")

    track_val = _safe_row_value(row, "track", "")
    name_val  = _safe_row_value(row, "name", item.get("name", code)) or code

    # ─ DART 이벤트 (non-blocking; 실패해도 기존 결과 유지) ────────
    kr_event_context: dict = {}
    try:
        from screener.events import fetch_kr_dart_events
        kr_event_context = fetch_kr_dart_events(code)
    except Exception as _exc:
        kr_event_context = {
            "code": code, "events": [], "event_score": None,
            "event_flags": [], "positive_events": [], "negative_events": [],
            "neutral_events": [], "error": str(_exc), "source": "DART",
        }

    # 이벤트 조정 shadow score (on-demand 전용; 기존 score 필드 불변)
    _kr_ev_adj = _calc_event_adjusted_score(
        base_score=float(_safe_row_value(row, "validated_score") or 0.0),
        event_context=kr_event_context,
        risk_score=_safe_row_value(row, "top_risk_score"),
        market="KR",
    )

    return {
        "ticker":                  code,
        "row":                     row,
        "track":                   str(track_val) if track_val else "",
        "rr_score":                rr,
        "exp":                     exp,
        "fund":                    {},
        "analyst":                 {},
        "upside":                  upside,
        "_ondemand":               True,
        "_screener_comparison":    screener_comparison,
        "_screener_p75":           screener_p75,
        "_screener_median":        screener_median,
        "_market":                 item.get("market", "KOSPI"),
        "_name":                   str(name_val),
        "kr_event_context":        kr_event_context,
        "event_adjusted_score":    _kr_ev_adj["event_adjusted_score"],
        "event_delta":             _kr_ev_adj["event_delta"],
        "event_adjustment_reason": _kr_ev_adj["event_adjustment_reason"],
        "event_score_source":      "DART",
    }, ""


def _ui_has_value(value) -> bool:
    if value is None:
        return False
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return str(value).strip() not in ("", "nan", "None", "—")


def _ui_float(value, default: float = float("nan")) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _fmt_score(value) -> str:
    v = _ui_float(value)
    return "—" if not math.isfinite(v) else f"{v:.0f}"


def _fmt_pct(value) -> str:
    v = _ui_float(value)
    return "—" if not math.isfinite(v) else f"{v:+.1f}%"


def _ticker_deep_dive_verdict(row: dict, rr_score: float) -> tuple[str, str]:
    validated = _ui_float(row.get("validated_score"))
    final = _ui_float(row.get("final_score"))
    risk = _ui_float(row.get("top_risk_score"), 0)
    buyable = _ui_float(row.get("buyable_score"))
    breakout = _ui_float(row.get("breakout_score"))
    catalyst = _ui_float(row.get("catalyst_score"))
    ret20 = _ui_float(row.get("ret_20d"), 0)
    track = str(row.get("track", "") or "")
    risk_flag = str(row.get("risk_flag", "") or "")

    if not any(math.isfinite(v) for v in [validated, final, rr_score, buyable, breakout, catalyst]):
        return "Insufficient Data", "#64748b"
    if risk >= 90 or "Avoid" in track or "Weak" in track:
        return "Avoid", "#f87171"
    if risk >= 75 or risk_flag:
        return "Watch", "#fbbf24"
    if ret20 >= 35 and buyable < 55:
        return "Too Extended", "#f59e0b"
    if validated >= 65 and final >= 60 and rr_score >= 60 and (breakout >= 50 or catalyst >= 55):
        return "Bullish", "#4ade80"
    if validated >= 55 or final >= 55 or rr_score >= 60:
        return "Watch", "#fbbf24"
    return "Insufficient Data", "#64748b"


def _render_ticker_deep_dive(found: dict, market: str):
    row_obj = found.get("row", {})
    row = dict(row_obj) if hasattr(row_obj, "items") else {}
    ticker = found.get("ticker", row.get("ticker", ""))
    name = row.get("company", row.get("name", ticker))
    track = str(found.get("track", row.get("track", "")) or "—")
    rr_score = _ui_float(found.get("rr_score"))
    fd = found.get("fund", {}) or {}
    an = found.get("analyst", {}) or {}

    validated = row.get("validated_score")
    final = row.get("final_score")
    leader_final = row.get("leader_final")
    buyable_final = row.get("buyable_final")
    risk = row.get("top_risk_score")
    risk_flag = str(row.get("risk_flag", "") or "")
    ret5 = row.get("ret_5d")
    ret20 = row.get("ret_20d")
    volume = _ui_float(row.get("volume_score"))
    buyable = _ui_float(row.get("buyable_score"))
    breakout = _ui_float(row.get("breakout_score"))
    catalyst = _ui_float(row.get("catalyst_score"))
    validated_f = _ui_float(validated)
    final_f = _ui_float(final)
    risk_f = _ui_float(risk)
    ret20_f = _ui_float(ret20)

    verdict, verdict_color = _ticker_deep_dive_verdict(row, rr_score)

    missing_notes = []
    if not an:
        missing_notes.append("No analyst data available")
    if market == "KR" and not _ui_has_value(row.get("supply_score")):
        missing_notes.append("No supply data available")

    notes = []
    if verdict == "Bullish":
        notes.append("Backtest-aligned score, live adjusted score, and re-rating score are all constructive.")
    elif verdict == "Too Extended":
        notes.append("Momentum is strong, but the 20-day move looks extended relative to the buyable setup.")
    elif verdict == "Avoid":
        notes.append("Risk or track status points to an avoid/weak setup.")
    elif verdict == "Watch":
        notes.append("Score stack is mixed enough to keep this on watch rather than treating it as a clean entry.")
    else:
        notes.append("Not enough score fields are present to form a strong UI verdict.")

    if validated_f >= 60:
        notes.append("validated_score supports the name on the backtest-aligned ranking path.")
    elif math.isfinite(validated_f):
        notes.append("validated_score is not yet strong enough to carry the setup alone.")

    if final_f > validated_f + 5:
        notes.append("final_score is above validated_score, suggesting live overlays are helping the setup.")
    elif math.isfinite(final_f) and math.isfinite(validated_f) and final_f < validated_f - 5:
        notes.append("final_score is below validated_score, suggesting live overlays are discounting the setup.")

    if risk_f >= 80 or risk_flag:
        notes.append("Risk is elevated; review ret_5d, ret_20d, and risk_flag before acting.")
    elif math.isfinite(risk_f):
        notes.append("Risk score is not the main blocker in the current row.")

    if breakout >= 60 or catalyst >= 60:
        notes.append("Breakout/catalyst fields point to an active setup rather than a purely passive watch.")
    elif math.isfinite(breakout) or math.isfinite(catalyst):
        notes.append("Breakout/catalyst confirmation is not especially strong yet.")

    if math.isfinite(volume) and volume < 40:
        notes.append("Weak-volume warning: volume_score is low in the selected row.")
    if ret20_f >= 35 and buyable < 55:
        notes.append("Overextension warning: strong 20-day return but buyable_score is below the buyable gate.")

    notes.extend(missing_notes)
    notes = notes[:6]

    score_items = [
        ("Validated Score", "Backtest-aligned", validated),
        ("Live Adjusted", "final_score", final),
        ("Re-rating", "UI narrative", rr_score),
        ("Leader Aggregate", "legacy/track", leader_final),
        ("Buyable Aggregate", "legacy/track", buyable_final),
    ]
    setup_items = [
        ("Track", track),
        ("Leader", _fmt_score(row.get("leader_score"))),
        ("Momentum", _fmt_score(row.get("momentum_score"))),
        ("Breakout", _fmt_score(row.get("breakout_score"))),
        ("Volume", _fmt_score(row.get("volume_score"))),
        ("Buyable", _fmt_score(row.get("buyable_score"))),
        ("Catalyst", _fmt_score(row.get("catalyst_score"))),
    ]
    risk_items = [
        ("Risk", _fmt_score(risk)),
        ("Risk flag", risk_flag or "—"),
        ("5D", _fmt_pct(ret5)),
        ("20D", _fmt_pct(ret20)),
    ]

    score_html = "".join(
        f'<div style="background:rgba(0,0,0,0.22);border-radius:7px;padding:8px;text-align:center">'
        f'<div style="font-size:10px;color:#64748b">{label}</div>'
        f'<div style="font-size:18px;font-weight:900;color:#f1f5f9">{_fmt_score(value)}</div>'
        f'<div style="font-size:10px;color:#475569">{sub}</div></div>'
        for label, sub, value in score_items
    )
    setup_html = "".join(
        f'<div><span style="color:#64748b">{label}</span><br>'
        f'<span style="font-weight:800;color:#f1f5f9">{value}</span></div>'
        for label, value in setup_items
    )
    risk_html = "".join(
        f'<div><span style="color:#64748b">{label}</span><br>'
        f'<span style="font-weight:800;color:#f1f5f9">{value}</span></div>'
        for label, value in risk_items
    )
    note_html = "".join(f'<li>{note}</li>' for note in notes)

    st.markdown(
        f'<div style="background:#0b1220;border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:10px;padding:16px;margin:12px 0">'
        f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">'
        f'<div><div style="font-size:13px;color:#64748b">Ticker Deep Dive</div>'
        f'<div style="font-size:19px;font-weight:900;color:#f1f5f9">{ticker} | {name}</div></div>'
        f'<div style="background:{verdict_color}22;color:{verdict_color};border:1px solid {verdict_color}55;'
        f'border-radius:6px;padding:5px 10px;font-size:13px;font-weight:900">{verdict}</div></div>'
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:14px">{score_html}</div>'
        f'<div style="display:grid;grid-template-columns:1.2fr 0.8fr;gap:14px;margin-top:14px">'
        f'<div><div style="font-size:12px;font-weight:900;color:#94a3b8;margin-bottom:6px">Setup</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:12px">{setup_html}</div></div>'
        f'<div><div style="font-size:12px;font-weight:900;color:#94a3b8;margin-bottom:6px">Risk</div>'
        f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;font-size:12px">{risk_html}</div></div></div>'
        f'<div style="margin-top:14px"><div style="font-size:12px;font-weight:900;color:#94a3b8;margin-bottom:5px">Narrative note</div>'
        f'<ul style="margin:0;padding-left:18px;color:#cbd5e1;font-size:12px;line-height:1.55">{note_html}</ul></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    major_fields = [
        "validated_score", "final_score", "rr_score", "leader_final", "buyable_final",
        "leader_score", "momentum_score", "breakout_score", "volume_score",
        "buyable_score", "catalyst_score", "top_risk_score", "risk_flag",
        "ret_5d", "ret_20d", "supply_score", "fund_score", "analyst_score",
    ]
    with st.expander("Data availability", expanded=False):
        availability = []
        for field in major_fields:
            if field == "rr_score":
                value = rr_score
            elif field == "fund_score":
                value = fd.get("fund_score")
            elif field == "analyst_score":
                value = an.get("analyst_score")
            else:
                value = row.get(field)
            availability.append({
                "field": field,
                "status": "present" if _ui_has_value(value) else "missing",
            })
        st.dataframe(pd.DataFrame(availability), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# 후보 통합 (전 트랙 → Re-rating Score 정렬)
# ════════════════════════════════════════════════════════════════════

def build_candidates(market: str = "US", track_filter: str = "전체",
                     exclude_turnaround: bool = False,
                     sort_by: str = "validated_score") -> list[dict]:
    """
    4개 트랙을 통합해서 정렬.
    sort_by: validated_score(기본) / rr_score / final_score / breakout_score / volume_score
    """
    if market == "US":
        keys     = ["us_turnaround","us_leader","us_breakout","us_buyable"]
        analyst  = st.session_state.get("us_analyst",{})
        fund_map = st.session_state.get("us_fund",{})
    else:
        keys     = ["kr_turnaround","kr_leader","kr_breakout","kr_buyable"]
        analyst  = {}
        fund_map = st.session_state.get("kr_fund",{})

    seen = set(); rows = []
    for key in keys:
        df = st.session_state.get(key)
        if df is None or isinstance(df, tuple): continue
        try:
            if df.empty: continue
        except Exception:
            continue
        for ticker, row in df.iterrows():
            if ticker in seen: continue
            track = str(row.get("track","") or "")
            # 트랙 필터
            if track_filter != "전체":
                if track_filter == "Leader"     and "Leader"     not in track: continue
                if track_filter == "Breakout"   and "Breakout"   not in track: continue
                if track_filter == "Turnaround" and "Turnaround" not in track: continue
                if track_filter == "Buyable"    and "Buyable"    not in track and "Pullback" not in track: continue
            # Turnaround는 메인 랭킹에서 제외 (별도 섹션에서 표시)
            if exclude_turnaround and "Turnaround" in track: continue

            # 한국 ETF 제외 (KODEX, TIGER 등)
            if market == "KR":
                name = str(row.get("name","") or "")
                if is_kr_candidate_etf_name(name):
                    continue

            # Risk >= 90 + 추세 훼손 = 제외
            risk = float(row.get("top_risk_score",0) or 0)
            ret5 = float(row.get("ret_5d",0) or 0)
            flag = str(row.get("risk_flag","") or "")
            if risk >= 90 and ret5 < 0 and ("음봉" in flag or "윗꼬리" in flag):
                continue

            fd = fund_map.get(ticker, {})
            an = analyst.get(ticker, {})
            row_dict = dict(row)
            if an: row_dict.update(an)

            rr_score = calc_rerating_score(row_dict, fd, an)
            exp      = estimate_return(row_dict)

            seen.add(ticker)
            # 업사이드 계산 (애널리스트 목표가 우선, 없으면 예상수익률)
            upside = float(an.get("upside_pct", float("nan")) or float("nan"))
            if not math.isfinite(upside):
                upside = float(exp.get("expected_mid", float("nan")) or float("nan"))

            rows.append({
                "ticker":     ticker,
                "row":        row,
                "track":      track,
                "rr_score":   rr_score,
                "exp":        exp,
                "fund":       fd,
                "analyst":    an,
                "upside":     upside,   # 업사이드 % (표시용)
            })

    # 정렬 키 선택 (validated_score 기본 → rr_score → final_score 순)
    def _sort_key(c):
        if sort_by == "validated_score":
            return float(c.get("row",{}).get("validated_score") or c.get("rr_score",0) or 0)
        elif sort_by == "rr_score":
            return float(c.get("rr_score",0) or 0)
        elif sort_by == "final_score":
            return float(c.get("row",{}).get("final_score") or c.get("rr_score",0) or 0)
        elif sort_by == "breakout_score":
            return float(c.get("row",{}).get("breakout_score",0) or 0)
        elif sort_by == "volume_score":
            return float(c.get("row",{}).get("volume_score",0) or 0)
        return float(c.get("rr_score",0) or 0)
    rows.sort(key=_sort_key, reverse=True)
    return rows


# ════════════════════════════════════════════════════════════════════
# UI 헬퍼
# ════════════════════════════════════════════════════════════════════

TRACK_META = {
    "Hot Leader / Extended":   {"color":"#f59e0b","icon":"🔥","short":"Leader"},
    "Hot Leader / Buyable":    {"color":"#f59e0b","icon":"🔥","short":"Leader"},
    "Breakout Signal":         {"color":"#a78bfa","icon":"⚡","short":"Breakout"},
    "Leader / Buyable":        {"color":"#4ade80","icon":"✅","short":"Buyable"},
    "Leader / Pullback Wait":  {"color":"#60a5fa","icon":"🔵","short":"Pullback"},
    "Turnaround / Early":      {"color":"#60a5fa","icon":"🔄","short":"Turnaround"},
    "Watch Only":              {"color":"#64748b","icon":"👁","short":"Watch"},
    "Avoid / Weak":            {"color":"#334155","icon":"❌","short":"Avoid"},
}

def _fp(v, d=1, suffix="%"):
    try: v=float(v)
    except: return "—"
    if not math.isfinite(v): return "—"
    return f"{'+'if v>=0 else ''}{v:.{d}f}{suffix}"

def _pc(v):
    try: v=float(v)
    except: return "#64748b"
    if not math.isfinite(v): return "#64748b"
    return "#4ade80" if v>=0 else "#f87171"

def _sc(s):
    try: s=float(s)
    except: return "#64748b"
    return "#f59e0b" if s>=75 else "#818cf8" if s>=60 else "#64748b"

def _bar(score, w=70):
    try: sc=max(0,min(100,float(score)))
    except: sc=50
    c=_sc(sc); fw=int(w*sc/100)
    return (f'<div style="display:flex;align-items:center;gap:5px">'
            f'<div style="width:{w}px;height:6px;background:#0f172a;border-radius:3px;overflow:hidden">'
            f'<div style="width:{fw}px;height:6px;background:{c};border-radius:3px"></div></div>'
            f'<span style="font-size:12px;font-weight:900;color:{c}">{sc:.0f}</span></div>')

def _track_badge(track):
    meta = TRACK_META.get(track, {"color":"#64748b","icon":"?","short":track[:8]})
    c = meta["color"]
    return (f'<span style="background:{c}22;color:{c};border:1px solid {c}44;'
            f'padding:2px 8px;border-radius:4px;font-size:11px;font-weight:800">'
            f'{meta["icon"]} {meta["short"]}</span>')


# ════════════════════════════════════════════════════════════════════
# 종목 카드 (MU식 리레이팅 중심)
# ════════════════════════════════════════════════════════════════════

def render_candidate_card(rank: int, c: dict, market: str):
    ticker  = c["ticker"]
    row     = c["row"]
    rr      = c["rr_score"]
    track   = c["track"]
    fd      = c["fund"]
    an      = c["analyst"]
    exp     = c["exp"]
    meta    = TRACK_META.get(track, {"color":"#64748b","icon":"?","short":"?"})
    tc      = meta["color"]

    # 기본 정보
    name    = row.get("company", row.get("name", ticker))
    price   = float(row.get("price", 0) or 0)
    ret5    = float(row.get("ret_5d",  0) or 0)
    ret20   = float(row.get("ret_20d", 0) or 0)
    rs_pct  = float(row.get("rs_rank_pct", 50) or 50)
    brk     = float(row.get("breakout_score", 0) or 0)
    cat     = float(row.get("catalyst_score",50) or 50)
    vol     = float(row.get("volume_score",  50) or 50)
    risk    = float(row.get("top_risk_score", 0) or 0)
    flag    = str(row.get("risk_flag","") or "")
    is_theme= bool(row.get("is_theme_pick", False))
    theme_name = str(row.get("theme_name","") or "")
    ldr_f   = float(row.get("leader_final", 50) or 50)

    # 예상 수익률
    exp_mid  = float(exp.get("expected_mid",  0) or 0)
    exp_low  = float(exp.get("expected_low",  0) or 0)
    exp_high = float(exp.get("expected_high", 0) or 0)
    exp_conf = exp.get("confidence", "참고용")
    bt_qual  = exp.get("bt_quality", "none")
    conf_icon= "✓" if bt_qual=="high" else "~" if bt_qual=="medium" else "?"
    conf_c   = "#4ade80" if bt_qual=="high" else "#fbbf24" if bt_qual=="medium" else "#64748b"
    show_exp = exp_conf != "참고용"

    # 재무 요약
    fund_grade = str(fd.get("fund_grade","") or "")
    rev_growth = float(fd.get("rev_growth", float("nan")) or float("nan"))
    rev_accel  = bool(fd.get("rev_accel", False))

    # 애널리스트
    eps_rev    = an.get("eps_revision","neutral")
    upside_pct = float(an.get("upside_pct", float("nan")) or float("nan"))
    next_earn  = an.get("next_earnings","")
    dte        = an.get("days_to_earnings")

    # Re-rating Score 색상 (강조)
    rr_c = "#4ade80" if rr>=70 else "#fbbf24" if rr>=50 else "#f87171"

    # 위험 배지
    risk_badge = ""
    if risk >= 90:
        risk_badge = f'<span style="background:rgba(248,113,113,0.15);color:#f87171;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">⚠ Risk{risk:.0f}</span>'
    elif risk >= 75:
        risk_badge = f'<span style="background:rgba(251,191,36,0.15);color:#fbbf24;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">Risk{risk:.0f}</span>'

    # 실적 발표 임박
    earn_badge = ""
    if dte is not None and 0 <= dte <= 14:
        ec = "#f87171" if dte<=7 else "#fbbf24"
        earn_badge = f'<span style="background:{ec}22;color:{ec};padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">📅D-{dte}</span>'

    # 테마 배지
    theme_badge = f'<span style="background:rgba(245,158,11,0.15);color:#f59e0b;padding:1px 7px;border-radius:3px;font-size:11px;font-weight:700">🎯{theme_name}</span>' if is_theme and theme_name else ""

    # EPS 배지
    eps_badge = ""
    if eps_rev == "up":
        eps_badge = '<span style="background:rgba(74,222,128,0.15);color:#4ade80;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">▲EPS상향</span>'
    elif eps_rev == "down":
        eps_badge = '<span style="background:rgba(248,113,113,0.12);color:#f87171;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">▼EPS하향</span>'

    # 재무 배지
    fund_badge = ""
    if fund_grade in ("A+","A"):
        fund_badge = f'<span style="background:rgba(74,222,128,0.12);color:#4ade80;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">재무{fund_grade}</span>'
    elif fund_grade == "B+":
        fund_badge = f'<span style="background:rgba(129,140,248,0.12);color:#818cf8;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">재무{fund_grade}</span>'

    if rev_accel:
        fund_badge += '<span style="background:rgba(74,222,128,0.12);color:#4ade80;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">🚀성장가속</span>'

    # 링크
    if market == "KR":
        link_html = (f'<a href="{row.get("naver_link","#")}" target="_blank" style="font-size:11px;color:#4ade80;background:rgba(74,222,128,0.08);padding:2px 8px;border-radius:4px;border:1px solid rgba(74,222,128,0.2);text-decoration:none">네이버</a> '
                     f'<a href="{row.get("dart_link","#")}" target="_blank" style="font-size:11px;color:#94a3b8;background:rgba(255,255,255,0.04);padding:2px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.08);text-decoration:none">DART</a>')
    else:
        link_html = (f'<a href="{row.get("yf_link","#")}" target="_blank" style="font-size:11px;color:#60a5fa;background:rgba(96,165,250,0.08);padding:2px 8px;border-radius:4px;border:1px solid rgba(96,165,250,0.2);text-decoration:none">Yahoo</a> '
                     f'<a href="{row.get("news_link","#")}" target="_blank" style="font-size:11px;color:#94a3b8;background:rgba(255,255,255,0.04);padding:2px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.08);text-decoration:none">뉴스</a>')

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0f172a,#111827);'
        f'border:1px solid rgba(255,255,255,0.06);border-top:3px solid {tc};'
        f'border-radius:14px;padding:18px 20px;margin-bottom:14px">'

        # ── 헤더 ────────────────────────────────────────────────
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">'
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
        f'<div style="min-width:34px;height:34px;border-radius:8px;'
        f'background:{"linear-gradient(135deg,#f59e0b,#ef4444)" if rank==1 else "#1e293b"};'
        f'display:flex;align-items:center;justify-content:center;font-weight:900;color:white;font-size:{"16px" if rank==1 else "13px"}">'
        f'{"👑" if rank==1 else f"#{rank}"}</div>'
        f'<div>'
        f'<div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:4px">'
        f'<span style="font-size:{"22px" if rank<=3 else "19px"};font-weight:900;color:#f1f5f9">{ticker}</span>'
        f'<span style="font-size:12px;color:#475569">{str(name)[:22]}</span>'
        f'{_track_badge(track)}'
        + risk_badge + earn_badge + theme_badge + eps_badge + fund_badge +
        f'</div>'
        f'<div style="font-size:12px;color:#64748b">'
        f'{row.get("sector","")[:20]} · {row.get("industry","")[:20] if "industry" in row else ""}'
        f'</div></div></div>'

        # ── Re-rating Score + Validated Score ──────────────────────
        f'<div style="text-align:right">'
        f'<div title="rr_score: UI 전용 리레이팅/내러티브 점수입니다. 백테스트에는 쓰지 않습니다." style="font-size:10px;color:#475569;margin-bottom:2px">🎯 리레이팅(rr)</div>'
        f'<div style="font-size:36px;font-weight:900;color:{rr_c};line-height:1">{rr:.0f}</div>'
        f'<div style="font-size:10px;color:{conf_c}">'
        + (f'{conf_icon} 예상 {exp_mid:+.0f}% ({exp_conf})' if show_exp else '? (BT없음)')
        + f'</div>'
        # validated_score (백테스트 검증 점수)
        + (lambda vs: f'<div title="validated_score: 백테스트와 공유하는 검증/기본 정렬 점수입니다." style="font-size:10px;color:#64748b;margin-top:2px">검증(validated) {vs:.0f}</div>'
           if vs > 0 else ''
          )(float(row.get("validated_score", 0) or 0))
        + f'</div></div></div>'

        # ── 핵심 지표 그리드 ─────────────────────────────────────
        f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-bottom:12px">',
        unsafe_allow_html=True,
    )

    metrics = [
        ("5일", _fp(ret5), _pc(ret5)),
        ("20일", _fp(ret20), _pc(ret20)),
        (f"RS({rs_pct:.0f}%)", _bar(rs_pct), None),
        ("Leader(legacy)", _bar(ldr_f), None),
        ("Breakout", _bar(brk), None),
        ("Catalyst", _bar(cat), None),
        ("Volume", _bar(vol), None),
    ]
    metric_html = ""
    for label, val, color in metrics:
        if color:
            metric_html += (
                f'<div style="background:rgba(0,0,0,0.2);border-radius:7px;padding:7px;text-align:center">'
                f'<div style="font-size:10px;color:#334155;margin-bottom:3px">{label}</div>'
                f'<div style="font-size:13px;font-weight:700;color:{color}">{val}</div></div>'
            )
        else:
            metric_html += (
                f'<div style="background:rgba(0,0,0,0.2);border-radius:7px;padding:7px;text-align:center">'
                f'<div style="font-size:10px;color:#334155;margin-bottom:3px">{label}</div>'
                f'{val}</div>'
            )

    st.markdown(metric_html + '</div>', unsafe_allow_html=True)

    # ── 상세 expander ─────────────────────────────────────────────
    with st.expander("📊 상세 검증  |  🎯 점수 기여도", expanded=False):
        # ── 점수 기여도 ──────────────────────────────────────────
        from screener.backtest import calc_score_explain
        explain = calc_score_explain(
            dict(row),
            weights=__import__("screener.backtest",fromlist=["load_optimal_weights"]).load_optimal_weights(),
            analyst=an,
            fund=fd,
        )
        exp_comps = explain.get("components",[])
        if exp_comps:
            st.markdown('<div style="font-size:13px;font-weight:800;color:#fbbf24;margin-bottom:6px">🎯 리레이팅 점수 분해</div>', unsafe_allow_html=True)
            items_html = f'<div style="background:rgba(0,0,0,0.3);border-radius:8px;padding:10px 14px;margin-bottom:10px">'
            items_html += f'<div style="font-size:11px;color:#475569;margin-bottom:4px">기준 점수 {explain["base"]:.0f}점</div>'
            for comp in sorted(exp_comps, key=lambda x: abs(x["delta"]), reverse=True):
                delta=comp["delta"]; c="#4ade80" if delta>0 else "#f87171"
                bar_w=min(80,int(abs(delta)*3))
                items_html += (
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">'
                    f'<div style="min-width:120px;font-size:11px;color:#94a3b8">{comp["label"]}</div>'
                    f'<div style="width:{bar_w}px;height:4px;background:{c};border-radius:2px"></div>'
                    f'<div style="font-size:12px;font-weight:700;color:{c}">{delta:+.1f}</div>'
                    f'<div style="font-size:10px;color:#475569">{comp.get("reason","")}</div>'
                    f'</div>'
                )
            items_html += f'<div style="border-top:1px solid rgba(255,255,255,0.06);margin-top:6px;padding-top:6px;font-size:13px;font-weight:900;color:#fbbf24">합계: {explain["total"]:.0f}점</div>'
            items_html += '</div>'
            st.markdown(items_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:12px;color:#334155;margin-bottom:8px">점수 기여도 데이터 없음 (점수가 중립 50점 근처)</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📈 리레이팅 근거**")
            reasons = []
            if eps_rev == "up": reasons.append(f"✅ EPS 컨센서스 상향 (+{float(an.get('eps_trend',0) or 0):.1f}%)")
            if rev_accel: reasons.append("✅ 매출 성장 가속")
            if math.isfinite(rev_growth) and rev_growth >= 20: reasons.append(f"✅ 매출 YoY {rev_growth:+.0f}%")
            if is_theme: reasons.append(f"✅ 테마 포함: {theme_name}")
            if brk >= 60: reasons.append(f"✅ 돌파 신호 ({brk:.0f})")
            if rs_pct >= 70: reasons.append(f"✅ RS 상위 {rs_pct:.0f}%")
            if not reasons: reasons.append("— 강한 리레이팅 근거 없음")
            for r in reasons: st.markdown(f'<div style="font-size:12px;color:#94a3b8">{r}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("**⚠️ 부담 요인**")
            burdens = []
            if risk >= 75: burdens.append(f"⚠ 과열 (Risk {risk:.0f})")
            dist20 = float(row.get("dist_20dma", float("nan")) or float("nan"))
            if math.isfinite(dist20) and dist20 >= 15: burdens.append(f"⚠ MA20 이격 {dist20:.0f}%")
            pe = float(fd.get("forward_pe", float("nan")) or float("nan"))
            if math.isfinite(pe) and pe >= 60: burdens.append(f"⚠ FwdPE {pe:.0f}x (고평가)")
            if eps_rev == "down": burdens.append("⚠ EPS 하향 중")
            if flag: burdens.append(f"⚠ {flag}")
            if not burdens: burdens.append("— 특이 부담 없음")
            for b in burdens: st.markdown(f'<div style="font-size:12px;color:#94a3b8">{b}</div>', unsafe_allow_html=True)

        with col3:
            st.markdown("**💡 판단**")
            rr_int = int(rr)
            if rr_int >= 70:
                st.markdown('<div style="font-size:12px;color:#4ade80">🔥 강한 리레이팅 후보</div>', unsafe_allow_html=True)
            elif rr_int >= 55:
                st.markdown('<div style="font-size:12px;color:#fbbf24">~ 중간 수준 후보</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:12px;color:#64748b">— 추세/촉매 약함</div>', unsafe_allow_html=True)

            if show_exp:
                exp_c = "#4ade80" if exp_mid>=10 else "#fbbf24" if exp_mid>=0 else "#f87171"
                st.markdown(
                    f'<div style="font-size:12px;color:{exp_c}">예상 1M: {exp_mid:+.0f}% ({exp_low:+.0f}~{exp_high:+.0f}%)</div>',
                    unsafe_allow_html=True,
                )

            # 애널리스트 목표가
            if math.isfinite(upside_pct):
                uc = "#4ade80" if upside_pct>=15 else "#fbbf24" if upside_pct>=0 else "#f87171"
                st.markdown(
                    f'<div style="font-size:12px;color:{uc}">목표가 업사이드: {upside_pct:+.0f}%</div>',
                    unsafe_allow_html=True,
                )
            if next_earn:
                st.markdown(f'<div style="font-size:11px;color:#475569">실적 발표: {next_earn}</div>', unsafe_allow_html=True)

        # MA 위치 + 링크
        ma_pairs = [("20MA","above_20dma"), ("50MA" if market=="US" else "60MA", "above_50dma" if market=="US" else "above_60dma"), ("200MA","above_200dma")]
        ma_html = " ".join(
            f'<span style="font-size:11px;padding:1px 6px;border-radius:10px;'
            f'background:{"rgba(74,222,128,0.1)" if row.get(k) else "rgba(248,113,113,0.08)"};'
            f'color:{"#4ade80" if row.get(k) else "#f87171"}">{"✓" if row.get(k) else "✗"} {lb}</span>'
            for lb,k in ma_pairs
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;flex-wrap:wrap;gap:6px">'
            f'<div>{ma_html}</div>'
            f'<div style="display:flex;gap:6px">{link_html}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 메인 탭 1: 🎯 MU식 리레이팅 후보
# ════════════════════════════════════════════════════════════════════

def _cash_reco(regime: dict, market: str) -> dict:
    """
    시장국면(0~100, 높을수록 건강)으로 현금 권고 산출.
    - cash_pct: 권장 현금비중(%)
    - buy_frac: 현금보다 나은(매수 검토) 후보 비율 — 컷라인 위치 계산용
    - label/color/signal
    """
    if not regime:
        return {"score": None, "cash_pct": 30, "buy_frac": 0.6,
                "label": "국면 미상 — 데이터 없음", "color": "#64748b", "signal": "중립"}
    key = "us_regime" if market == "US" else "kr_regime"
    s = float(regime.get(key, 50))
    # 건강할수록 현금 적게, 매수 후보 비율 높게
    cash_pct = int(round(max(10, min(90, 90 - s * 0.9))))     # s=100→0~, s=0→90
    buy_frac = max(0.0, min(1.0, (s - 20) / 60))               # s<=20 전량현금, s>=80 대부분매수
    if s >= 65:
        label, color, signal = "강세 — 적극 비중 확대 가능", "#10b981", "매수우위"
    elif s >= 50:
        label, color, signal = "중립~양호 — 선별 매수", "#84cc16", "선별매수"
    elif s >= 35:
        label, color, signal = "약세 — 신규 진입 신중, 상위만", "#f59e0b", "방어"
    else:
        label, color, signal = "위험 — 현금 비중 높게, 매도 우선", "#ef4444", "현금우위"
    return {"score": s, "cash_pct": cash_pct, "buy_frac": buy_frac,
            "label": label, "color": color, "signal": signal}


def render_rerating_tab(market: str):
    # 강한 테마 상단 요약
    sector_data   = st.session_state.get("sector_analysis",[])
    us_themes_uni = st.session_state.get("universe_themes_us",[])
    kr_themes_uni = st.session_state.get("universe_themes_kr",[])
    hot_sectors   = [s["sector"] for s in sector_data if s.get("hot")][:5]
    hot_themes_us = [t["theme"] for t in us_themes_uni if t.get("hot")][:4]
    hot_themes_kr = [t["theme"] for t in kr_themes_uni if t.get("hot")][:4]
    hot_themes    = hot_themes_us if market=="US" else hot_themes_kr

    if hot_sectors or hot_themes:
        pills_html = ""
        for s in hot_sectors:
            pills_html += f'<span style="background:rgba(245,158,11,0.12);color:#f59e0b;padding:2px 9px;border-radius:12px;font-size:12px;font-weight:700;margin:2px">{s}</span>'
        for t in hot_themes:
            pills_html += f'<span style="background:rgba(129,140,248,0.12);color:#818cf8;padding:2px 9px;border-radius:12px;font-size:12px;font-weight:700;margin:2px">{t}</span>'
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:10px 14px;margin-bottom:14px">'
            f'<span style="font-size:12px;color:#475569;margin-right:8px">🔥 강한 테마·섹터:</span>'
            f'{pills_html}</div>',
            unsafe_allow_html=True,
        )

    # 트랙 필터 + 정렬 기준
    c_filter, c_sort = st.columns([3,2])
    with c_filter:
        track_options = ["전체","Leader","Breakout","Turnaround","Buyable"]
        track_filter  = st.radio(
            "후보 필터", track_options, horizontal=True, key=f"track_filter_{market}",
            help="Leader/Buyable은 기존 트랙 분류입니다. Leader는 leader_final, Buyable은 buyable_final 성격의 트랙별 집계 점수와 연결됩니다.",
        )
    with c_sort:
        sort_options = {
            "검증점수 (validated_score: 백테스트 정렬)": "validated_score",
            "리레이팅점수 (rr_score: UI 참고)":    "rr_score",
            "라이브 보정점수 (final_score)":        "final_score",
            "돌파점수":                 "breakout_score",
            "거래량점수":               "volume_score",
        }
        sort_label = st.selectbox("정렬 기준", list(sort_options.keys()),
                                  key=f"sort_by_{market}",
                                  help="기본값은 validated_score입니다. final_score는 재무/테마/컨텍스트/회피 보정 후 점수이고, rr_score는 UI 전용 리레이팅 참고 점수입니다.")
        sort_by = sort_options[sort_label]
        st.caption("validated_score=백테스트 정렬, final_score=라이브 보정, rr_score=UI 리레이팅 참고")

    main_candidates = build_candidates(market, track_filter,
                                       exclude_turnaround=True, sort_by=sort_by)

    if not main_candidates and not build_candidates(market, "Turnaround"):
        st.info("스크리너를 먼저 실행해주세요. 사이드바에서 실행 버튼을 누르세요.")
        # ─ 스크리너 실행 전에도 볼 수 있는 것들 ──────────────────────
        # 1) 내 보유 종목
        held = portfolio.get_holdings()
        if held:
            st.markdown("##### 💼 내 보유 종목")
            h_rows = [{
                "티커": tk, "종목명": h.get("name",""),
                "비중": h.get("weight_raw",""),
                "평단가": h.get("avg_price"),
                "메모": h.get("memo",""),
            } for tk, h in held.items()]
            st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
            st.caption("스크리너를 실행하면 이 종목들이 GPT 후보에 자동 포함됩니다.")

        # 2) 과거 저널 기록 (이전에 돌린 결과)
        past_dates = journal.list_dates(market)
        if past_dates:
            st.markdown("##### 📒 이전 기록 (과거 GPT 추천 & 채점)")
            psel = st.selectbox("기록 날짜 보기", past_dates, key=f"past_journal_{market}")
            pentry = journal.load(market, psel)
            if pentry:
                prows = journal.score_entry(pentry, eval_days=5)
                pdf = pd.DataFrame([{
                    "실제매수": "🛒" if portfolio.is_bought(market, psel, r["ticker"]) else "",
                    "순위": r.get("rank"), "티커": r["ticker"],
                    "종목": str(r["name"])[:12], "대응": r["action"],
                    "진입가": r.get("entry_close"), "수익%": r.get("ret"),
                    "판정": r.get("verdict"),
                    "현재수익%": r.get("now_ret"),
                    "핵심판단": str(r.get("note") or "")[:30],
                } for r in prows])
                st.dataframe(pdf, use_container_width=True, hide_index=True)
                st.caption(f"'{psel}' 기록 — 자세한 채점/실제매수 체크는 'GPT 리레이팅 분석' 탭에서.")
        elif not held:
            st.caption("아직 보유 종목도 과거 기록도 없습니다. '내 포트폴리오' 탭에서 보유 종목을 입력해보세요.")
        return

    # 상위 요약
    top3 = [f"#{i+1} {c['ticker']}({c['rr_score']:.0f})" for i,c in enumerate(main_candidates[:3])]
    st.markdown(
        f'<div style="font-size:13px;color:#475569;margin-bottom:12px">'
        f'{len(main_candidates)}개 후보 (Turnaround 제외) · Re-rating Score 순 · '
        f'<strong style="color:#f59e0b">{" / ".join(top3)}</strong></div>',
        unsafe_allow_html=True,
    )

    # ── 💵 시장국면 + 현금 배너 ───────────────────────────────────
    regime = st.session_state.get("regime_us" if market=="US" else "regime_kr")
    cash = _cash_reco(regime, market)
    _sc_txt = f"{cash['score']:.0f}/100" if cash["score"] is not None else "—"
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.03);border-left:4px solid {cash["color"]};'
        f'border-radius:8px;padding:10px 14px;margin-bottom:12px">'
        f'<span style="font-size:14px;font-weight:900;color:{cash["color"]}">💵 시장국면 {_sc_txt} · {cash["signal"]}</span>'
        f'<span style="font-size:12px;color:#475569"> — {cash["label"]} · 권장 현금비중 ~{cash["cash_pct"]}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 현금 컷라인 위치: 건강할수록 더 많은 종목이 현금 위로
    cash_cut = int(round(len(main_candidates) * cash["buy_frac"]))

    for rank, c in enumerate(main_candidates, 1):
        if rank - 1 == cash_cut:
            st.markdown(
                f'<div style="background:rgba(245,158,11,0.08);border:1px dashed {cash["color"]};'
                f'border-radius:8px;padding:8px 14px;margin:10px 0;text-align:center">'
                f'<span style="font-size:13px;font-weight:900;color:{cash["color"]}">'
                f'💵 현금 (CASH) — 여기보다 아래 순위는 현재 국면에서 "현금만 못함" → 신규 진입 보류 권장</span></div>',
                unsafe_allow_html=True,
            )
        render_candidate_card(rank, c, market)
    if cash_cut >= len(main_candidates) and main_candidates:
        st.caption("💵 현 국면에선 전 종목이 현금보다 나은 편 — 그래도 분할·손절 원칙 유지.")

    # Turnaround 별도 섹션 (메인 랭킹과 혼재 방지)
    ta_candidates = build_candidates(market, "Turnaround", exclude_turnaround=False)
    if ta_candidates and track_filter in ("전체","Turnaround"):
        st.markdown(
            '<div style="background:rgba(96,165,250,0.06);border:1px solid rgba(96,165,250,0.15);' +
            'border-radius:10px;padding:12px 16px;margin:20px 0 10px">' +
            '<div style="font-size:14px;font-weight:900;color:#60a5fa">🔄 Turnaround 후보 (저점 반등 초기)</div>' +
            '<div style="font-size:12px;color:#475569;margin-top:3px">' +
            'MU식 리레이팅(비싸도 더 갈 종목)과 다른 바구니. 싸서 반등 중인 종목. 메인 랭킹과 분리.</div></div>',
            unsafe_allow_html=True,
        )
        for rank, c in enumerate(ta_candidates[:6], 1):
            render_candidate_card(rank, c, market)

    # ── Shadow Event Score Comparison (on-demand 캐시 재사용) ─────────────
    _od_prefix = "ondemand_us_" if market == "US" else "ondemand_kr_"
    _shadow_rows = []
    for c in main_candidates[:20]:
        tk = c["ticker"]
        od_key   = f"{_od_prefix}{tk}"
        od_state = st.session_state.get(od_key)
        if not (od_state and od_state.get("found")):
            continue
        od       = od_state["found"]
        ev_adj   = od.get("event_adjusted_score")
        ev_delta = od.get("event_delta", 0.0) or 0.0
        ev_rsn   = od.get("event_adjustment_reason", "")
        nm       = str(c.get("company", c.get("name", tk)))[:16]
        vs       = c.get("validated_score", 0) or 0
        _shadow_rows.append({
            "ticker":            tk,
            "name":              nm,
            "validated_score":   round(float(vs), 1),
            "event_adj_score":   round(float(ev_adj), 1) if ev_adj is not None else None,
            "delta":             round(float(ev_delta), 1),
            "reason":            str(ev_rsn)[:45],
        })

    with st.expander(
        f"⚡ Shadow Event Score Comparison"
        f"{f' — {len(_shadow_rows)} tickers' if _shadow_rows else ''}",
        expanded=False,
    ):
        st.caption("Shadow comparison only; default screener ranking unchanged.")
        if _shadow_rows:
            _shadow_df = pd.DataFrame(_shadow_rows)
            st.dataframe(_shadow_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No event context loaded for current candidates. "
                "Run on-demand analysis for individual tickers first."
            )

    # ── Top-N Batch Shadow Event Enrichment (shadow-only) ────────────────
    _render_shadow_event_batch(market, main_candidates)
    # ── Shadow A/B Outcome Tracker (forward outcome tracking) ────────────
    _render_shadow_ab_tracker(market)
    # ── KR Segment Shadow Composite (shadow-only) ───────────────────────
    if market == "KR":
        _render_kr_segment_shadow(market, main_candidates)


def _render_kr_segment_shadow(market: str, main_candidates: list) -> None:
    """
    🧩 KR Segment Shadow Composite — KOSPI/KOSDAQ별 shadow 전용 합성 점수.
    SHADOW-ONLY: 실제 후보 랭킹/선정/백테스트 불변. 네트워크 호출 없음.
    최신 KR shadow event batch가 있으면 event-adjusted base로 재사용한다.
    """
    with st.expander("🧩 KR Segment Shadow Composite (shadow-only)", expanded=False):
        st.caption(
            "Shadow 전용 — KOSPI(추세/수급) vs KOSDAQ(돌파/거래량/이벤트리스크) "
            "세그먼트별 휴리스틱 합성 점수. 실제 랭킹/선정/백테스트와 무관. "
            "DART/EDGAR/Earnings 재호출 없음 — 기존 shadow batch 재사용."
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            limit = st.selectbox("Top-N", [20, 50, 100], index=1,
                                 key="kr_seg_shadow_limit")
        with c2:
            run = st.button("🧩 KR 세그먼트 shadow composite 계산",
                            key="kr_seg_shadow_run")

        if run:
            from screener.segment_shadow import apply_kr_segment_shadow_composite

            val_rows = []
            for c in main_candidates[:limit]:
                r = c.get("row")
                if r is None or not hasattr(r, "get"):
                    continue
                rec = dict(r)
                rec.setdefault("ticker", c["ticker"])
                if rec.get("validated_score") is None:
                    rec["validated_score"] = c.get("rr_score", 0)
                val_rows.append(rec)
            validated_df = pd.DataFrame(val_rows)
            shadow_df = st.session_state.get("shadow_event_batch_kr")
            if validated_df.empty:
                st.warning("KR 후보가 없습니다. 먼저 스크리너를 실행하세요.")
            else:
                seg_df = apply_kr_segment_shadow_composite(
                    validated_df, shadow_df=shadow_df, limit=limit)
                st.session_state["kr_segment_shadow_composite"] = seg_df

        seg_df = st.session_state.get("kr_segment_shadow_composite")
        if seg_df is None:
            st.caption("아직 계산 전. 위 버튼을 눌러 현재 KR 후보에 대한 세그먼트 shadow 점수를 계산하세요.")
            return
        if seg_df.empty:
            st.info("세그먼트 결과가 비어 있습니다.")
            return

        total   = len(seg_df)
        n_kospi = int((seg_df["kr_segment"] == "KOSPI").sum())
        n_kosdaq = int((seg_df["kr_segment"] == "KOSDAQ").sum())
        n_unknown = int((seg_df["kr_segment"] == "UNKNOWN").sum())
        n_up    = int((seg_df["kr_segment_candidate_action"] == "upgrade_watch").sum())
        n_down  = int((seg_df["kr_segment_candidate_action"] == "downgrade_watch").sum())
        st.markdown(
            f"**{total}개** · KOSPI {n_kospi} · KOSDAQ {n_kosdaq} · UNKNOWN {n_unknown} "
            f"· ⬆️ upgrade {n_up} · ⬇️ downgrade {n_down}"
        )

        display_cols = [
            "original_rank", "segment_shadow_rank", "segment_shadow_rank_delta",
            "ticker", "name", "kr_segment", "validated_score",
            "kr_segment_shadow_score", "kr_segment_shadow_delta",
            "kr_segment_candidate_action", "kr_segment_shadow_profile",
            "adverse_event_risk_level", "adverse_event_risk_score",
            "adverse_event_categories",
            "kr_segment_shadow_reason", "shadow_event_flags",
        ]
        present = [c for c in display_cols if c in seg_df.columns]
        disp = seg_df[present].copy()
        if "adverse_event_categories" in disp.columns:
            disp["adverse_event_categories"] = disp["adverse_event_categories"].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else str(v or "")
            )
        if "shadow_event_flags" in disp.columns:
            disp["shadow_event_flags"] = disp["shadow_event_flags"].apply(
                lambda v: " ".join(v) if isinstance(v, list) else str(v or "")
            )
        st.dataframe(disp, use_container_width=True, hide_index=True)

        m1, m2 = st.columns(2)
        with m1:
            st.caption("⬆️ Top 10 상승 (segment_shadow_rank_delta)")
            up = seg_df.sort_values("segment_shadow_rank_delta", ascending=False).head(10)
            st.dataframe(up[[c for c in ("ticker", "name", "kr_segment",
                                          "segment_shadow_rank_delta",
                                          "kr_segment_shadow_delta") if c in up.columns]],
                         use_container_width=True, hide_index=True)
        with m2:
            st.caption("⬇️ Top 10 하락 (segment_shadow_rank_delta)")
            down = seg_df.sort_values("segment_shadow_rank_delta", ascending=True).head(10)
            st.dataframe(down[[c for c in ("ticker", "name", "kr_segment",
                                            "segment_shadow_rank_delta",
                                            "kr_segment_shadow_delta") if c in down.columns]],
                         use_container_width=True, hide_index=True)
        st.caption("⚠️ Shadow-only 분석입니다. 실제 후보 랭킹은 변경되지 않았습니다.")


def _render_shadow_event_batch(market: str, main_candidates: list) -> None:
    """
    Top-N 후보를 DART/EDGAR/Earnings 컨텍스트로 일괄 enrich 해서
    validated_score 대비 event-adjusted shadow score / 잠재 순위 변화를 보여준다.

    SHADOW-ONLY. 실제 후보 랭킹/선정/백테스트에는 절대 영향 없음.
    네트워크 호출은 버튼 클릭 시에만 발생한다(페이지 렌더 중에는 호출 안 함).
    """
    _batch_key = "shadow_event_batch_us" if market == "US" else "shadow_event_batch_kr"

    with st.expander("⚡ Top-N Shadow Event Batch (shadow-only)", expanded=False):
        st.caption(
            "Shadow 전용 — 실제 후보 랭킹/선정/백테스트에 전혀 영향 없음. "
            "버튼을 눌러야만 DART/EDGAR/Earnings 네트워크 호출이 일어납니다."
        )
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            limit = st.selectbox("Top-N", [20, 50, 100], index=1,
                                 key=f"shadow_batch_limit_{market}")
        with c2:
            force = st.checkbox("강제 새로고침", value=False,
                                key=f"shadow_batch_force_{market}")
        with c3:
            run = st.button("⚡ Top-N Shadow Event Batch 실행",
                            key=f"shadow_batch_run_{market}")

        if run:
            from screener.event_shadow import enrich_shadow_events_for_candidates

            def _cv(c, key, default=None):
                r = c.get("row")
                if r is None or not hasattr(r, "get"):
                    return default
                v = r.get(key, default)
                return default if v is None else v

            cand_rows = []
            for c in main_candidates[:limit]:
                cand_rows.append({
                    "ticker":          c["ticker"],
                    "name":            str(_cv(c, "name", c["ticker"]))[:20],
                    "validated_score": float(_cv(c, "validated_score",
                                                  c.get("rr_score", 0)) or 0),
                    "top_risk_score":  float(_cv(c, "top_risk_score", 0) or 0),
                    "market":          str(_cv(c, "market", "")),
                })
            cand_df = pd.DataFrame(cand_rows)

            if cand_df.empty:
                st.warning("후보가 없습니다. 먼저 스크리너를 실행하세요.")
            else:
                with st.spinner(f"Top-{limit} shadow 이벤트 수집 중..."):
                    res = enrich_shadow_events_for_candidates(
                        cand_df, market, limit=limit, force_refresh=force,
                    )
                st.session_state[_batch_key] = res

        res = st.session_state.get(_batch_key)
        if res is None:
            st.caption(
                "아직 실행 전. 위 버튼을 눌러 현재 Top-N 후보의 shadow 이벤트를 수집하세요. "
                "(페이지를 새로 열어도 자동 호출되지 않음)"
            )
            return
        if res.empty:
            st.info("배치 결과가 비어 있습니다.")
            return

        # 요약
        n         = len(res)
        n_err     = int((res["shadow_event_status"] == "error").sum())
        n_missing = int(res["shadow_event_status"].isin(["missing", "unavailable"]).sum())
        n_up      = int((res["shadow_candidate_action"] == "upgrade_watch").sum())
        n_down    = int((res["shadow_candidate_action"] == "downgrade_watch").sum())
        st.markdown(
            f"**{n}개 enrich** · ⬆️ upgrade_watch {n_up} · ⬇️ downgrade_watch {n_down} "
            f"· ⚠️ error {n_err} · ◻️ missing {n_missing}"
        )

        display_cols = [
            "original_rank", "shadow_rank", "shadow_rank_delta", "ticker", "name",
            "validated_score", "shadow_event_adjusted_score", "shadow_event_delta",
            "shadow_event_flags", "shadow_candidate_action", "shadow_event_status",
            "shadow_event_reason",
        ]
        if market == "KR":
            display_cols.insert(5, "kr_market_segment")
        present = [c for c in display_cols if c in res.columns]
        disp = res[present].copy()
        if "shadow_event_flags" in disp.columns:
            disp["shadow_event_flags"] = disp["shadow_event_flags"].apply(
                lambda v: " ".join(v) if isinstance(v, list) else str(v or "")
            )
        st.dataframe(disp, use_container_width=True, hide_index=True)

        m1, m2 = st.columns(2)
        with m1:
            st.caption("⬆️ Top 10 상승 (shadow_rank_delta)")
            up = res.sort_values("shadow_rank_delta", ascending=False).head(10)
            st.dataframe(up[[c for c in ("ticker", "name", "shadow_rank_delta",
                                          "shadow_event_delta") if c in up.columns]],
                         use_container_width=True, hide_index=True)
        with m2:
            st.caption("⬇️ Top 10 하락 (shadow_rank_delta)")
            down = res.sort_values("shadow_rank_delta", ascending=True).head(10)
            st.dataframe(down[[c for c in ("ticker", "name", "shadow_rank_delta",
                                            "shadow_event_delta") if c in down.columns]],
                         use_container_width=True, hide_index=True)
        st.caption("⚠️ Shadow-only 분석입니다. 실제 후보 랭킹은 변경되지 않았습니다.")

        # ── 📌 Shadow A/B 스냅샷 저장 ──────────────────────────────
        st.divider()
        if st.button("📌 Shadow A/B 스냅샷 저장", key=f"shadow_ab_save_{market}"):
            from screener.shadow_outcomes import (
                build_shadow_ab_snapshot, append_shadow_ab_snapshot,
            )

            def _cv2(c, key, default=None):
                r = c.get("row")
                if r is None or not hasattr(r, "get"):
                    return default
                v = r.get(key, default)
                return default if v is None else v

            top_n = len(res)
            val_rows = []
            for c in main_candidates[:top_n]:
                val_rows.append({
                    "ticker":          c["ticker"],
                    "name":            str(_cv2(c, "name", c["ticker"]))[:20],
                    "validated_score": float(_cv2(c, "validated_score",
                                                   c.get("rr_score", 0)) or 0),
                    "price":           _cv2(c, "price"),
                    "market":          str(_cv2(c, "market", "")),
                })
            validated_df = pd.DataFrame(val_rows)
            # KR: attach the latest segment composite df so snapshot rows carry
            # kr_segment_shadow_score/_delta/_profile/_candidate_action.
            segment_df = None
            if market == "KR":
                segment_df = st.session_state.get("kr_segment_shadow_composite")
            snap = build_shadow_ab_snapshot(validated_df, res, market, top_n=top_n,
                                            segment_df=segment_df)
            status = append_shadow_ab_snapshot(snap)
            if status.get("status") == "added":
                st.success(
                    f"스냅샷 저장됨: `{snap['snapshot_id']}` — "
                    f"overlap {snap['summary']['overlap_count']} / "
                    f"added {snap['summary']['added_count']} / "
                    f"removed {snap['summary']['removed_count']}"
                )
            elif status.get("status") == "duplicate":
                st.info(f"이미 저장된 스냅샷입니다: `{snap['snapshot_id']}`")
            else:
                st.warning(f"스냅샷 저장 실패: {status.get('error')}")


def _render_shadow_ab_tracker(market: str) -> None:
    """
    📊 Shadow A/B Outcome Tracker — 저장된 스냅샷과 (있으면) outcome 결과를 표시.
    SHADOW-ONLY / measurement-only. 자동 네트워크 호출 없음.
    forward outcome tracking이며 historical backtest가 아님.
    """
    from screener.shadow_outcomes import (
        list_shadow_ab_snapshots, list_shadow_outcome_results,
    )

    with st.expander("📊 Shadow A/B Outcome Tracker", expanded=False):
        st.caption(
            "Forward outcome tracking (NOT a historical backtest). "
            "오늘의 shadow 선택을 기록하고, 5/10/20 거래일 경과 후 실현 수익률로 평가합니다. "
            "실제 후보 랭킹/선정/백테스트와 무관 — 측정 전용."
        )
        snaps = list_shadow_ab_snapshots(market=market, limit=20)
        if not snaps:
            st.info("저장된 스냅샷이 없습니다. 위에서 batch 실행 후 '📌 Shadow A/B 스냅샷 저장'을 누르세요.")
            return

        snap_rows = []
        for s in snaps:
            sm = s.get("summary", {})
            snap_rows.append({
                "snapshot_id":  s.get("snapshot_id"),
                "asof":         s.get("asof_date"),
                "top_n":        s.get("top_n"),
                "overlap":      sm.get("overlap_count"),
                "added":        sm.get("added_count"),
                "removed":      sm.get("removed_count"),
                "avg_delta":    sm.get("avg_shadow_delta"),
                "up_watch":     sm.get("upgrade_watch_count"),
                "down_watch":   sm.get("downgrade_watch_count"),
            })
        st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)

        if market == "KR" and snaps:
            seg = snaps[0].get("summary", {}).get("by_segment")
            if seg:
                st.caption(
                    "최근 스냅샷 세그먼트: "
                    + " · ".join(f"{k} {v}" for k, v in seg.items())
                )

        # outcome 평가 (US만 기본 fetcher 지원; 데이터 없으면 missing 표시)
        sel = st.selectbox(
            "스냅샷 선택 (outcome 평가/조회)",
            [s["snapshot_id"] for s in snaps],
            key=f"shadow_ab_sel_{market}",
        )

        # ── 스냅샷 세그먼트 attribution (KR, outcome 이전 카운트) ─────────
        if market == "KR":
            from screener.shadow_outcomes import summarize_segment_snapshot_attribution
            _sel_snap = next((s for s in snaps if s["snapshot_id"] == sel), None)
            _att = summarize_segment_snapshot_attribution(_sel_snap) if _sel_snap else {}
            if _att.get("has_segment_fields"):
                cnt   = _att["counts"]
                seg_c = cnt["by_segment"]
                act_c = cnt["by_action"]
                sa    = cnt["by_segment_action"]
                st.markdown(
                    "**세그먼트 구성** — "
                    + " · ".join(f"{k} {v}" for k, v in seg_c.items())
                )
                st.caption(
                    "프로필: " + " · ".join(f"{k} {v}" for k, v in cnt["by_profile"].items())
                    + "  |  액션: " + " · ".join(f"{k} {v}" for k, v in act_c.items())
                )
                st.caption(
                    f"KOSPI ⬆️{sa.get('KOSPI|upgrade_watch', 0)} ⬇️{sa.get('KOSPI|downgrade_watch', 0)}"
                    f"  ·  KOSDAQ ⬆️{sa.get('KOSDAQ|upgrade_watch', 0)} ⬇️{sa.get('KOSDAQ|downgrade_watch', 0)}"
                )

        if market == "US":
            cc1, cc2 = st.columns(2)
            with cc1:
                cost_bps = st.number_input("비용(bps)", value=10.0, step=5.0,
                                           key=f"shadow_ab_cost_{market}")
            with cc2:
                slip_bps = st.number_input("슬리피지(bps)", value=5.0, step=5.0,
                                           key=f"shadow_ab_slip_{market}")
            if st.button("현재 가능한 가격 데이터로 outcome 평가",
                         key=f"shadow_ab_eval_{market}"):
                from screener.shadow_outcomes import (
                    evaluate_snapshot_outcomes, append_shadow_outcome_result,
                    default_price_fetcher,
                )
                snap = next((s for s in snaps if s["snapshot_id"] == sel), None)
                if snap:
                    with st.spinner("가격 데이터로 outcome 평가 중..."):
                        result = evaluate_snapshot_outcomes(
                            snap, default_price_fetcher,
                            benchmark_fetcher=default_price_fetcher,
                            cost_bps=cost_bps, slippage_bps=slip_bps,
                        )
                        append_shadow_outcome_result(result)
                    vt = result["results"]["validated_top"]["20"]
                    if vt.get("count_available", 0) == 0:
                        st.info(
                            "아직 평가 가능한 가격 데이터가 부족합니다 "
                            "(5/10/20 거래일 경과 후 다시 시도하세요)."
                        )
                    else:
                        st.success("outcome 평가 완료 — 아래 결과 표 참조")
        else:
            st.caption(
                "Outcome evaluation requires price data after 5/10/20 trading days. "
                "(KR 기본 fetcher는 미지원 — DI 평가 프레임워크는 테스트로 검증됨.)"
            )

        # 저장된 outcome 결과 표시
        results = list_shadow_outcome_results(snapshot_id=sel)
        if results:
            r = results[0]
            st.markdown(f"**Outcome — `{sel}`** (평가시각 {r.get('evaluated_at')})")
            out_rows = []
            for grp, byh in r.get("results", {}).items():
                if grp == "shadow_minus_validated":
                    continue
                for h, m in byh.items():
                    if not isinstance(m, dict):
                        continue
                    out_rows.append({
                        "group":     grp,
                        "horizon":   h,
                        "avg_ret%":  None if m.get("avg_return") is None
                                     else round(m["avg_return"] * 100, 2),
                        "win%":      None if m.get("win_rate") is None
                                     else round(m["win_rate"] * 100, 1),
                        "hit5%":     None if m.get("hit_rate_5pct") is None
                                     else round(m["hit_rate_5pct"] * 100, 1),
                        "n":         m.get("count_available"),
                        "missing":   m.get("count_missing"),
                    })
            if out_rows:
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

            # ── 세그먼트 outcome attribution (있을 때만) ───────────────
            seg_attr = r.get("results", {}).get("segment_attribution") or {}
            if seg_attr:
                st.markdown("**🧩 세그먼트 outcome attribution** (KOSPI vs KOSDAQ · 프로필 · 액션)")

                def _attr_df(section: dict):
                    rows_ = []
                    for grp, byh in section.items():
                        for h, m in byh.items():
                            if not isinstance(m, dict):
                                continue
                            rows_.append({
                                "group":    grp,
                                "horizon":  h,
                                "avg_ret%": None if m.get("avg_return") is None
                                            else round(m["avg_return"] * 100, 2),
                                "win%":     None if m.get("win_rate") is None
                                            else round(m["win_rate"] * 100, 1),
                                "hit5%":    None if m.get("hit_rate_5pct") is None
                                            else round(m["hit_rate_5pct"] * 100, 1),
                                "worst%":   None if m.get("worst_return") is None
                                            else round(m["worst_return"] * 100, 2),
                                "n":        m.get("count_available"),
                                "miss":     m.get("count_missing"),
                                "seg_Δ":    m.get("avg_kr_segment_shadow_delta"),
                                "ev_Δ":     m.get("avg_shadow_event_delta"),
                            })
                    return pd.DataFrame(rows_)

                for label, key in (("by segment", "by_segment"),
                                   ("by profile", "by_profile"),
                                   ("by action", "by_action"),
                                   ("by segment/action", "by_segment_action")):
                    section = seg_attr.get(key) or {}
                    if section:
                        st.caption(label)
                        st.dataframe(_attr_df(section),
                                     use_container_width=True, hide_index=True)
        else:
            st.caption("아직 이 스냅샷의 outcome 결과가 없습니다.")


# ════════════════════════════════════════════════════════════════════
# 탭 2: 🤖 GPT 리레이팅 분석
# ════════════════════════════════════════════════════════════════════

def render_gpt_analysis_tab(market: str):
    st.markdown(
        '<div style="background:rgba(129,140,248,0.07);border:1px solid rgba(129,140,248,0.2);'
        'border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#94a3b8">'
        '두 가지 순위가 <strong style="color:#f1f5f9">반드시 다릅니다</strong>.<br>'
        '📊 <strong>저평가</strong>: 적정가 대비 할인율 — LOGI가 1등일 수 있음<br>'
        '🔥 <strong>리레이팅</strong>: 추정치·테마·수급이 계속 올라가서 비싸도 더 비싸질 종목 — AMD가 1등일 수 있음<br>'
        '<span style="color:#475569">모드를 고르면 목적에 맞는 프롬프트가 자동 생성됩니다.</span></div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "GPT 분석 모드",
        ["🔍 급등 후보 점검 (단기)", "🔥 리레이팅 중심 (MU식)", "📊 저평가 중심", "⚖️ 균형형"],
        horizontal=True,
        key=f"gpt_mode_{market}",
    )

    # ════════════════════════════════════════════════════════════════════
    # 🔎 즉시 점수 조회 & GPT 분석 (새로운 기능)
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div style="font-size:13px;font-weight:900;color:#f1f5f9">🔎 특정 종목 즉시 조회</div>', unsafe_allow_html=True)
    st.caption("Ticker lookup searches loaded screener results only; it does not query an external/live source. Load latest saved results if you do not want to rerun the screener.")
    
    ticker_input = st.text_input(
        "티커 입력 (예: 005930 또는 000660)",
        placeholder="종목코드 6자리 또는 미국 티커",
        key=f"ticker_search_{market}"
    )
    
    if ticker_input and ticker_input.strip():
        tk_clean = ticker_input.strip().upper()
        search_col1, search_col2 = st.columns([1, 1])
        
        with search_col1:
            if st.button("⚡ 점수 계산", use_container_width=True, key=f"btn_calc_{market}_{tk_clean}"):
                st.session_state[f"ticker_lookup_query_{market}"] = tk_clean
                st.session_state[f"ticker_lookup_action_{market}"] = "score"
        
        with search_col2:
            if st.button("🤖 GPT에 물어보기", use_container_width=True, key=f"btn_gpt_{market}_{tk_clean}"):
                st.session_state[f"ticker_lookup_query_{market}"] = tk_clean
                st.session_state[f"ticker_lookup_action_{market}"] = "gpt"
        
        lookup_query = st.session_state.get(f"ticker_lookup_query_{market}", "")
        lookup_action = st.session_state.get(f"ticker_lookup_action_{market}", "")
        lookup_found = _find_ticker_candidate(lookup_query, market) if lookup_query else None
        
        # 점수 계산 결과 표시
        if lookup_action == "score" and lookup_query:
            try:
                found = lookup_found
                
                if found:
                    tk = found["ticker"]
                    row = found["row"]
                    name = row.get("company", row.get("name", tk))
                    rr_score = found.get("rr_score", 0)
                    track = found.get("track", "")
                    
                    # 카드 표시
                    st.markdown(f"""
                    <div style="background:#0f172a;border:2px solid #818cf8;border-radius:10px;padding:16px;margin:10px 0">
                    <div style="font-size:16px;font-weight:900;color:#f1f5f9">{tk} | {name}</div>
                    <div style="font-size:12px;color:#94a3b8;margin:8px 0">{row.get('sector','')}</div>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px">
                        <div style="text-align:center">
                            <div title="rr_score: UI 전용 리레이팅/내러티브 점수입니다. 백테스트에는 쓰지 않습니다." style="font-size:11px;color:#475569">리레이팅(rr)</div>
                            <div style="font-size:24px;font-weight:900;color:#818cf8">{rr_score:.0f}</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:11px;color:#475569">현재가</div>
                            <div style="font-size:20px;font-weight:700;color:#f1f5f9">{row.get('price','?'):,.0f}</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:11px;color:#475569">20일 수익률</div>
                            <div style="font-size:20px;font-weight:700;color:#{'4ade80' if float(row.get('ret_20d',0) or 0)>0 else 'f87171'}">{float(row.get('ret_20d',0) or 0):+.1f}%</div>
                        </div>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                    _render_ticker_deep_dive(found, market)
                else:
                    if market == "US":
                        st.info(
                            f"**{lookup_query}** 을(를) 현재 로드된 스크리너 결과에서 찾을 수 없습니다. "
                            "기존 스크리너 결과는 그대로 유지됩니다."
                        )
                        od_key = f"ondemand_us_{lookup_query}"
                        if st.button(
                            f"🔄 {lookup_query} On-demand 분석 (외부 데이터 수집)",
                            key=f"btn_od_{lookup_query}",
                        ):
                            with st.spinner(f"{lookup_query} 데이터 수집 중 (5~15초)..."):
                                od_found, od_err = _analyze_us_ticker_ondemand(lookup_query)
                            st.session_state[od_key] = {"found": od_found, "err": od_err}

                        od_state = st.session_state.get(od_key)
                        if od_state:
                            od_err   = od_state.get("err", "")
                            od_found = od_state.get("found")
                            if od_err:
                                st.error(f"On-demand 분석 실패: {od_err}")
                            elif od_found:
                                od_row   = od_found["row"]
                                od_tk    = od_found["ticker"]
                                od_name  = _safe_row_value(od_row, "company", od_tk) or od_tk
                                od_rr    = float(_safe_row_value(od_found, "rr_score", 0.0))
                                od_ret20 = float(_safe_row_value(od_row, "ret_20d", 0.0))
                                od_comp  = od_found.get("_screener_comparison")
                                od_p75   = od_found.get("_screener_p75")
                                od_med   = od_found.get("_screener_median")
                                od_vs    = float(_safe_row_value(od_row, "validated_score", 0.0))
                                od_prox  = _safe_row_value(od_row, "week52_prox")
                                od_surge = _safe_row_value(od_row, "vol_surge")
                                od_rflag = str(_safe_row_value(od_row, "risk_flag", "") or "")

                                comp_map = {
                                    "strong":  ("스크리너 상위 25% 이상", "#4ade80"),
                                    "average": ("스크리너 중간권",        "#fbbf24"),
                                    "weak":    ("스크리너 하위권",         "#f87171"),
                                }
                                comp_text, comp_color = comp_map.get(od_comp, ("비교 불가", "#64748b"))
                                p75_txt = (
                                    f" (P75={od_p75:.0f} / 중앙={od_med:.0f})"
                                    if od_p75 is not None else ""
                                )

                                st.markdown(
                                    f'<div style="font-size:11px;color:#64748b;margin:6px 0">'
                                    f'On-demand 분석 결과 (스크리너 외 신규 조회)&nbsp;|&nbsp;'
                                    f'<span style="color:{comp_color};font-weight:700">'
                                    f'{comp_text}</span>{p75_txt}</div>',
                                    unsafe_allow_html=True,
                                )

                                prox_str = (
                                    f"{float(od_prox):.0f}%"
                                    if od_prox is not None and math.isfinite(float(od_prox))
                                    else "—"
                                )
                                surge_str = (
                                    f"{float(od_surge):.2f}x"
                                    if od_surge is not None and math.isfinite(float(od_surge))
                                    else "—"
                                )
                                rflag_html = (
                                    f'<div style="margin-top:8px;color:#f87171;font-size:12px;'
                                    f'font-weight:700">⚠ {od_rflag}</div>'
                                    if od_rflag else ""
                                )

                                st.markdown(f"""
<div style="background:#0f172a;border:2px solid #818cf8;border-radius:10px;padding:16px;margin:10px 0">
<div style="font-size:16px;font-weight:900;color:#f1f5f9">{od_tk} | {od_name}</div>
<div style="font-size:12px;color:#94a3b8;margin:4px 0">{od_row.get('sector','')}&nbsp;|&nbsp;{od_row.get('track','')}</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px">
  <div style="text-align:center"><div style="font-size:11px;color:#475569">validated</div>
  <div style="font-size:22px;font-weight:900;color:#818cf8">{od_vs:.0f}</div></div>
  <div style="text-align:center"><div style="font-size:11px;color:#475569">리레이팅(rr)</div>
  <div style="font-size:22px;font-weight:900;color:#f1f5f9">{od_rr:.0f}</div></div>
  <div style="text-align:center"><div style="font-size:11px;color:#475569">20일 수익률</div>
  <div style="font-size:22px;font-weight:900;color:#{'4ade80' if od_ret20 > 0 else 'f87171'}">{od_ret20:+.1f}%</div></div>
  <div style="text-align:center"><div style="font-size:11px;color:#475569">52주 고점 근접</div>
  <div style="font-size:20px;font-weight:900;color:#f1f5f9">{prox_str}</div></div>
</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px;font-size:11px;color:#94a3b8">
  <div>거래량 배율: <b style="color:#f1f5f9">{surge_str}</b></div>
  <div>5일: <b style="color:#f1f5f9">{float(_safe_row_value(od_row, 'ret_5d', 0.0)):+.1f}%</b></div>
  <div>60일: <b style="color:#f1f5f9">{float(_safe_row_value(od_row, 'ret_60d', 0.0)):+.1f}%</b></div>
</div>
{rflag_html}
</div>
""", unsafe_allow_html=True)
                                _render_ticker_deep_dive(od_found, "US")
                                # ── EDGAR 이벤트 카드 ───────────────────────
                                ev_us      = od_found.get("us_event_context") or {}
                                ev_us_score = ev_us.get("event_score")
                                ev_us_error = ev_us.get("error")
                                if ev_us_error:
                                    st.caption(f"📋 SEC EDGAR 이벤트: {ev_us_error}")
                                elif ev_us_score is not None:
                                    ev_color = (
                                        "#4ade80" if ev_us_score >= 60
                                        else "#f87171" if ev_us_score <= 40
                                        else "#fbbf24"
                                    )
                                    st.markdown(
                                        f'<div style="background:#0f172a;border:1px solid {ev_color};'
                                        f'border-radius:8px;padding:12px;margin:8px 0">'
                                        f'<div style="font-size:12px;font-weight:700;color:#94a3b8">'
                                        f'📋 Real Event Catalyst (SEC EDGAR)</div>'
                                        f'<div style="font-size:20px;font-weight:900;color:{ev_color};margin:4px 0">'
                                        f'Event Score: {ev_us_score}</div>',
                                        unsafe_allow_html=True,
                                    )
                                    for flag in ev_us.get("event_flags", [])[:5]:
                                        st.markdown(
                                            f'<div style="font-size:11px;color:#cbd5e1;margin:2px 0">{flag}</div>',
                                            unsafe_allow_html=True,
                                        )
                                    ev_recent = ev_us.get("events", [])[:3]
                                    if ev_recent:
                                        st.markdown(
                                            '<div style="font-size:11px;color:#64748b;margin-top:6px">최근 공시:</div>',
                                            unsafe_allow_html=True,
                                        )
                                        for ev_item in ev_recent:
                                            cls_icon = (
                                                "✅" if ev_item.get("classification") == "positive"
                                                else "⚠" if ev_item.get("classification") == "negative"
                                                else "·"
                                            )
                                            st.markdown(
                                                f'<div style="font-size:11px;color:#94a3b8;margin:1px 0">'
                                                f'{cls_icon} {ev_item.get("filingDate","")} '
                                                f'[{ev_item.get("form","")}] '
                                                f'{str(ev_item.get("description",""))[:50]}</div>',
                                                unsafe_allow_html=True,
                                            )
                                    st.markdown("</div>", unsafe_allow_html=True)
                                # ── Shadow Score card ──────────────────────
                                ev_adj_sc = od_found.get("event_adjusted_score")
                                ev_delta  = od_found.get("event_delta", 0.0) or 0.0
                                ev_reason = od_found.get("event_adjustment_reason", "")
                                if ev_adj_sc is not None:
                                    d_color = (
                                        "#4ade80" if ev_delta > 0
                                        else "#f87171" if ev_delta < 0
                                        else "#fbbf24"
                                    )
                                    d_sign = "+" if ev_delta >= 0 else ""
                                    # verdict note
                                    _track = od_found.get("track", "") or ""
                                    if ev_delta >= 5 and "Avoid" not in _track:
                                        verdict_note = "Event-confirmed Bullish"
                                        v_color = "#4ade80"
                                    elif ev_delta <= -8:
                                        verdict_note = "Watch / Event Risk"
                                        v_color = "#f87171"
                                    else:
                                        verdict_note = ""
                                        v_color = ""
                                    verdict_html = (
                                        f'<span style="color:{v_color};font-weight:700;'
                                        f'font-size:11px;margin-left:8px">{verdict_note}</span>'
                                        if verdict_note else ""
                                    )
                                    st.markdown(
                                        f'<div style="background:#0a1628;border:1px dashed #334155;'
                                        f'border-radius:7px;padding:10px 12px;margin:6px 0">'
                                        f'<div style="font-size:11px;color:#475569;margin-bottom:4px">'
                                        f'⚡ Event-adjusted Shadow Score</div>'
                                        f'<span style="font-size:22px;font-weight:900;color:#818cf8">'
                                        f'{ev_adj_sc:.0f}</span>'
                                        f'<span style="font-size:14px;color:{d_color};'
                                        f'font-weight:700;margin-left:8px">{d_sign}{ev_delta:.0f}</span>'
                                        f'{verdict_html}'
                                        f'<div style="font-size:10px;color:#475569;margin-top:4px">'
                                        f'{ev_reason}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(
                                        "Shadow score only — default screener ranking unchanged"
                                    )
                                # ── Earnings Catalyst card ───────────────
                                ec       = od_found.get("us_earnings_context") or {}
                                ec_score = ec.get("earnings_score")
                                ec_flags = ec.get("earnings_flags", [])
                                ec_date  = ec.get("latest_earnings_date")
                                ec_error = ec.get("error")
                                ec_eps   = ec.get("eps_surprise_pct")
                                if ec_error and ec_score is None:
                                    st.caption(f"📊 실적 데이터: {ec_error}")
                                elif ec_score is not None:
                                    ec_color = (
                                        "#4ade80" if ec_score >= 60
                                        else "#f87171" if ec_score <= 40
                                        else "#fbbf24"
                                    )
                                    date_lbl = f" — {ec_date}" if ec_date else ""
                                    flag_html = "".join(
                                        f'<div style="font-size:11px;margin:2px 0">{f}</div>'
                                        for f in ec_flags
                                    ) or '<div style="font-size:11px;color:#475569">서프라이즈 데이터 없음</div>'
                                    st.markdown(
                                        f'<div style="background:#0c1428;border:1px solid #1e293b;'
                                        f'border-radius:8px;padding:10px 12px;margin:6px 0">'
                                        f'<div style="font-size:11px;font-weight:700;color:#94a3b8;'
                                        f'margin-bottom:6px">📊 Earnings Catalyst{date_lbl}</div>'
                                        f'<div style="display:flex;align-items:center;gap:14px">'
                                        f'<div style="text-align:center;min-width:50px">'
                                        f'<div style="font-size:9px;color:#475569">실적점수</div>'
                                        f'<div style="font-size:24px;font-weight:900;color:{ec_color}">'
                                        f'{ec_score}</div></div>'
                                        f'<div style="flex:1">{flag_html}</div></div></div>',
                                        unsafe_allow_html=True,
                                    )
                    elif market == "KR" and _is_kr_query(lookup_query):
                        st.info(
                            f"**{lookup_query}** 을(를) 현재 로드된 스크리너 결과에서 찾을 수 없습니다. "
                            "한국 종목은 새로 조회해 on-demand 분석할 수 있습니다. "
                            "기존 스크리너 결과는 그대로 유지됩니다."
                        )
                        od_kr_key = f"ondemand_kr_{lookup_query}"
                        if st.button(
                            f"🔄 {lookup_query} On-demand KR 분석 (Naver 데이터 수집)",
                            key=f"btn_od_kr_{lookup_query}",
                        ):
                            with st.spinner(f"{lookup_query} Naver 데이터 수집 중 (10~30초)..."):
                                od_kr_found, od_kr_err = _analyze_kr_ticker_ondemand(lookup_query)
                            st.session_state[od_kr_key] = {"found": od_kr_found, "err": od_kr_err}

                        od_kr_state = st.session_state.get(od_kr_key)
                        if od_kr_state:
                            od_kr_err   = od_kr_state.get("err", "")
                            od_kr_found = od_kr_state.get("found")
                            if od_kr_err:
                                st.error(f"On-demand KR 분석 실패: {od_kr_err}")
                            elif od_kr_found:
                                od_row   = od_kr_found["row"]
                                od_tk    = od_kr_found["ticker"]
                                od_name  = od_kr_found.get("_name") or _safe_row_value(od_row, "name", od_tk) or od_tk
                                od_mkt   = od_kr_found.get("_market", "KOSPI")
                                od_rr    = float(_safe_row_value(od_kr_found, "rr_score", 0.0))
                                od_ret20 = float(_safe_row_value(od_row, "ret_20d", 0.0))
                                od_comp  = od_kr_found.get("_screener_comparison")
                                od_p75   = od_kr_found.get("_screener_p75")
                                od_med   = od_kr_found.get("_screener_median")
                                od_vs    = float(_safe_row_value(od_row, "validated_score", 0.0))
                                od_prox  = _safe_row_value(od_row, "week52_prox")
                                od_surge = _safe_row_value(od_row, "vol_surge")
                                od_sup   = _safe_row_value(od_row, "supply_score")
                                od_rflag = str(_safe_row_value(od_row, "risk_flag", "") or "")

                                comp_map = {
                                    "strong":  ("스크리너 상위 25% 이상", "#4ade80"),
                                    "average": ("스크리너 중간권",        "#fbbf24"),
                                    "weak":    ("스크리너 하위권",         "#f87171"),
                                }
                                comp_text, comp_color = comp_map.get(od_comp, ("비교 불가", "#64748b"))
                                p75_txt = (
                                    f" (P75={od_p75:.0f} / 중앙={od_med:.0f})"
                                    if od_p75 is not None else ""
                                )
                                prox_str = (
                                    f"{float(od_prox):.0f}%"
                                    if od_prox is not None and math.isfinite(float(od_prox))
                                    else "—"
                                )
                                surge_str = (
                                    f"{float(od_surge):.2f}x"
                                    if od_surge is not None and math.isfinite(float(od_surge))
                                    else "—"
                                )
                                sup_str = (
                                    f"{float(od_sup):.0f}"
                                    if od_sup is not None and math.isfinite(float(od_sup))
                                    else "—"
                                )
                                rflag_html = (
                                    f'<div style="margin-top:8px;color:#f87171;font-size:12px;'
                                    f'font-weight:700">⚠ {od_rflag}</div>'
                                    if od_rflag else ""
                                )

                                st.markdown(
                                    f'<div style="font-size:11px;color:#64748b;margin:6px 0">'
                                    f'On-demand KR 분석 ({od_mkt})&nbsp;|&nbsp;'
                                    f'<span style="color:{comp_color};font-weight:700">'
                                    f'{comp_text}</span>{p75_txt}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(f"""
<div style="background:#0f172a;border:2px solid #818cf8;border-radius:10px;padding:16px;margin:10px 0">
<div style="font-size:16px;font-weight:900;color:#f1f5f9">{od_tk} | {od_name}</div>
<div style="font-size:12px;color:#94a3b8;margin:4px 0">{od_mkt}&nbsp;|&nbsp;{_safe_row_value(od_row,'track','')}</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px">
  <div style="text-align:center"><div style="font-size:11px;color:#475569">validated</div>
  <div style="font-size:22px;font-weight:900;color:#818cf8">{od_vs:.0f}</div></div>
  <div style="text-align:center"><div style="font-size:11px;color:#475569">리레이팅(rr)</div>
  <div style="font-size:22px;font-weight:900;color:#f1f5f9">{od_rr:.0f}</div></div>
  <div style="text-align:center"><div style="font-size:11px;color:#475569">20일 수익률</div>
  <div style="font-size:22px;font-weight:900;color:#{'4ade80' if od_ret20 > 0 else 'f87171'}">{od_ret20:+.1f}%</div></div>
  <div style="text-align:center"><div style="font-size:11px;color:#475569">52주 고점 근접</div>
  <div style="font-size:20px;font-weight:900;color:#f1f5f9">{prox_str}</div></div>
</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px;font-size:11px;color:#94a3b8">
  <div>거래량 배율: <b style="color:#f1f5f9">{surge_str}</b></div>
  <div>수급점수: <b style="color:#f1f5f9">{sup_str}</b></div>
  <div>5일: <b style="color:#f1f5f9">{float(_safe_row_value(od_row, 'ret_5d', 0.0)):+.1f}%</b></div>
  <div>60일: <b style="color:#f1f5f9">{float(_safe_row_value(od_row, 'ret_60d', 0.0)):+.1f}%</b></div>
</div>
{rflag_html}
</div>
""", unsafe_allow_html=True)
                                _render_ticker_deep_dive(od_kr_found, "KR")

                                # ── DART 이벤트 카드 ──────────────────────────
                                ev       = od_kr_found.get("kr_event_context") or {}
                                ev_score = ev.get("event_score")
                                ev_error = ev.get("error")
                                ev_flags = ev.get("event_flags", [])
                                ev_recent = ev.get("events", [])[:3]

                                if ev_error:
                                    if "API 키 없음" in ev_error or "DART_API_KEY" in ev_error:
                                        st.caption(f"💡 {ev_error}")
                                    else:
                                        st.caption(f"DART 조회 오류: {ev_error}")
                                elif ev_score is not None:
                                    sc_color = (
                                        "#4ade80" if ev_score >= 60
                                        else "#f87171" if ev_score <= 40
                                        else "#fbbf24"
                                    )
                                    flag_html = "".join(
                                        f'<div style="font-size:11px;margin:2px 0">{f}</div>'
                                        for f in ev_flags
                                    ) or '<div style="font-size:11px;color:#475569">최근 30일 DART 공시 없음</div>'
                                    recent_html = "".join(
                                        f'<div style="font-size:10px;color:#64748b;margin:1px 0">'
                                        f'[{e["rcept_dt"]}] {e["report_nm"]}</div>'
                                        for e in ev_recent
                                    )
                                    st.markdown(f"""
<div style="background:#0c1428;border:1px solid #334155;border-radius:8px;padding:12px;margin:8px 0">
<div style="font-size:13px;font-weight:900;color:#94a3b8;margin-bottom:8px">📋 Real Event Catalyst (DART)</div>
<div style="display:flex;align-items:center;gap:16px">
  <div style="text-align:center;min-width:60px">
    <div style="font-size:10px;color:#475569">이벤트 점수</div>
    <div style="font-size:28px;font-weight:900;color:{sc_color}">{ev_score:.0f}</div>
  </div>
  <div style="flex:1">{flag_html}</div>
</div>
{f'<div style="margin-top:8px;border-top:1px solid #1e293b;padding-top:6px">{recent_html}</div>' if recent_html else ''}
</div>
""", unsafe_allow_html=True)
                                # ── Shadow Score card (KR) ─────────────────
                                kr_ev_adj_sc = od_kr_found.get("event_adjusted_score")
                                kr_ev_delta  = od_kr_found.get("event_delta", 0.0) or 0.0
                                kr_ev_reason = od_kr_found.get("event_adjustment_reason", "")
                                if kr_ev_adj_sc is not None:
                                    kr_d_color = (
                                        "#4ade80" if kr_ev_delta > 0
                                        else "#f87171" if kr_ev_delta < 0
                                        else "#fbbf24"
                                    )
                                    kr_d_sign = "+" if kr_ev_delta >= 0 else ""
                                    _kr_track = od_kr_found.get("track", "") or ""
                                    if kr_ev_delta >= 5 and "Avoid" not in _kr_track:
                                        kr_verdict = "Event-confirmed Bullish"
                                        kr_v_color = "#4ade80"
                                    elif kr_ev_delta <= -8:
                                        kr_verdict = "Watch / Event Risk"
                                        kr_v_color = "#f87171"
                                    else:
                                        kr_verdict = ""
                                        kr_v_color = ""
                                    kr_verdict_html = (
                                        f'<span style="color:{kr_v_color};font-weight:700;'
                                        f'font-size:11px;margin-left:8px">{kr_verdict}</span>'
                                        if kr_verdict else ""
                                    )
                                    st.markdown(
                                        f'<div style="background:#0a1628;border:1px dashed #334155;'
                                        f'border-radius:7px;padding:10px 12px;margin:6px 0">'
                                        f'<div style="font-size:11px;color:#475569;margin-bottom:4px">'
                                        f'⚡ Event-adjusted Shadow Score</div>'
                                        f'<span style="font-size:22px;font-weight:900;color:#818cf8">'
                                        f'{kr_ev_adj_sc:.0f}</span>'
                                        f'<span style="font-size:14px;color:{kr_d_color};'
                                        f'font-weight:700;margin-left:8px">{kr_d_sign}{kr_ev_delta:.0f}</span>'
                                        f'{kr_verdict_html}'
                                        f'<div style="font-size:10px;color:#475569;margin-top:4px">'
                                        f'{kr_ev_reason}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(
                                        "Shadow score only — DART screener ranking unchanged"
                                    )
                    else:
                        st.warning(
                            f"**{lookup_query}** 을(를) 현재 로드된 {market} 결과에서 찾을 수 없습니다. "
                            "기존 스크리너 결과는 그대로 유지됩니다."
                        )
            except Exception as e:
                st.error(f"점수 계산 오류: {e}")
        
        # GPT 분석 요청
        if lookup_action == "gpt" and lookup_query:
            try:
                found = lookup_found
                
                if found:
                    tk = found["ticker"]
                    row = found["row"]
                    an = found["analyst"]
                    fd = found["fund"]
                    _render_ticker_deep_dive(found, market)
                    nm = row.get("company", row.get("name", tk))
                    p = float(row.get("price", 0) or 0)
                    r20 = float(row.get("ret_20d", 0) or 0)
                    rs = float(row.get("rs_rank_pct", 50) or 50)
                    brk = float(row.get("breakout_score", 0) or 0)
                    cat = float(row.get("catalyst_score", 50) or 50)
                    rsk = float(row.get("top_risk_score", 0) or 0)
                    rr = found["rr_score"]
                    eps = an.get("eps_revision", "?")
                    fg = fd.get("fund_grade", "?")
                    cur = "원" if market == "KR" else "달러"
                    
                    sup_txt = ""
                    if market == "KR":
                        ssc = row.get("supply_score")
                        if isinstance(ssc, (int, float)) and math.isfinite(ssc):
                            sup_txt += f" | 수급점수{ssc:.0f}"
                    
                    stock_line = (f"{tk} | {str(nm)[:18]} | 현재가 {p:,.0f}{cur} | "
                                 f"20일{r20:+.1f}% | RS상위{rs:.0f}% | Breakout{brk:.0f} | "
                                 f"Catalyst{cat:.0f} | Risk{rsk:.0f} | EPS{eps} | "
                                 f"재무{fg} | 리레이팅점수{rr:.0f}{sup_txt}")
                    
                    # GPT 프롬프트 생성 (모드별)
                    mkt_name = "미국" if market == "US" else "한국"
                    
                    if "리레이팅" in mode:
                        prompt_body = f"""아래는 한 종목의 리레이팅 가능성 판단 요청입니다.

분석 대상:
{stock_line}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 목적]
이 종목이 MU처럼 비싸도 계속 리레이팅될 가능성이 있는가?

아래 5가지를 간결하게 분석해줘:

[Q1] 최근 컨센서스 추정치가 올라가는 중인가?
[Q2] EPS 성장률이 주가 상승을 정당화하는가?
[Q3] 테마·사이클의 지속성은?
[Q4] 경쟁사도 함께 상승하는가? (혼자만 오르는 건 아닌가?)
[Q5] 기관/애널리스트 목표가가 따라오는가?

[최종 판단]
- 리레이팅 가능성: 높음/중간/낮음
- 현금 대비 매력도: ★★★★★ / ★★★☆☆ / ★☆☆☆☆ 중 선택
- 위험요소 3가지
- 구체적 투자 전략 (진입/대기/회피)"""
                    else:
                        prompt_body = f"""아래는 {mkt_name} 종목 1개의 빠른 분석 요청입니다.

분석 대상:
{stock_line}

이 종목을 간결하게 평가해주세요:
1. 현재 상태 (추세, 촉매, 수급)
2. 단기 전망 (다음 2주)
3. 실전 대응 (진입/대기/회피)
4. 현금 대비 매력도 (★★★★★ ~ ★☆☆☆☆)
5. 주요 리스크"""
                    
                    st.markdown("**📋 생성된 GPT 프롬프트 (복사해서 GPT에 붙여넣으세요):**")
                    st.code(prompt_body, language="text")
                    
                    st.success("✅ 위 프롬프트를 복사하여 GPT에 붙여넣으면 됩니다!")
                else:
                    if market == "US":
                        od_key   = f"ondemand_us_{lookup_query}"
                        od_state = st.session_state.get(od_key)
                        if od_state and od_state.get("found"):
                            od_found = od_state["found"]
                            od_row   = od_found["row"]
                            _render_ticker_deep_dive(od_found, "US")
                            tk  = od_found["ticker"]
                            nm  = _safe_row_value(od_row, "company", tk) or tk
                            p   = float(_safe_row_value(od_row, "price", 0.0))
                            r20 = float(_safe_row_value(od_row, "ret_20d", 0.0))
                            rs  = float(_safe_row_value(od_row, "rs_rank_pct", 50.0))
                            brk = float(_safe_row_value(od_row, "breakout_score", 0.0))
                            cat = float(_safe_row_value(od_row, "catalyst_score", 50.0))
                            rsk = float(_safe_row_value(od_row, "top_risk_score", 0.0))
                            rr          = float(_safe_row_value(od_found, "rr_score", 0.0))
                            ev_us_ctx   = od_found.get("us_event_context") or {}
                            ev_us_sc    = ev_us_ctx.get("event_score")
                            ev_adj_sc   = od_found.get("event_adjusted_score")
                            ev_delta    = od_found.get("event_delta", 0.0) or 0.0
                            ev_us_txt   = f" | EDGAR이벤트{ev_us_sc:.0f}" if ev_us_sc is not None else ""
                            ev_adj_txt  = (
                                f" | 이벤트조정점수{ev_adj_sc:.0f}(delta{ev_delta:+.0f})"
                                if ev_adj_sc is not None else ""
                            )
                            ec_ctx     = od_found.get("us_earnings_context") or {}
                            ec_sc      = ec_ctx.get("earnings_score")
                            ec_eps_s   = ec_ctx.get("eps_surprise_pct")
                            earn_txt   = (
                                f" | 실적점수{ec_sc:.0f}"
                                + (f"(EPS{ec_eps_s:+.1f}%)" if ec_eps_s is not None else "")
                                if ec_sc is not None else ""
                            )
                            ev_pos_flags = ev_us_ctx.get("positive_events", [])[:2]
                            ev_neg_flags = ev_us_ctx.get("negative_events", [])[:2]
                            ev_flag_txt = ""
                            if ev_pos_flags:
                                ev_flag_txt += " 긍정:" + ",".join(
                                    f"[{e.get('form','')}]{str(e.get('description',''))[:20]}"
                                    for e in ev_pos_flags
                                )
                            if ev_neg_flags:
                                ev_flag_txt += " 리스크:" + ",".join(
                                    f"[{e.get('form','')}]{str(e.get('description',''))[:20]}"
                                    for e in ev_neg_flags
                                )
                            stock_line = (
                                f"{tk} | {str(nm)[:18]} | 현재가 ${p:,.2f} | "
                                f"20일{r20:+.1f}% | RS상위{rs:.0f}% | Breakout{brk:.0f} | "
                                f"Catalyst{cat:.0f} | Risk{rsk:.0f} | "
                                f"리레이팅점수{rr:.0f}{ev_us_txt}{ev_adj_txt}{earn_txt}{ev_flag_txt} [On-demand 분석]"
                            )
                            if "리레이팅" in mode:
                                prompt_body = f"""아래는 한 종목의 리레이팅 가능성 판단 요청입니다.

분석 대상:
{stock_line}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 목적]
이 종목이 MU처럼 비싸도 계속 리레이팅될 가능성이 있는가?

아래 5가지를 간결하게 분석해줘:

[Q1] 최근 컨센서스 추정치가 올라가는 중인가?
[Q2] EPS 성장률이 주가 상승을 정당화하는가?
[Q3] 테마·사이클의 지속성은?
[Q4] 경쟁사도 함께 상승하는가? (혼자만 오르는 건 아닌가?)
[Q5] 기관/애널리스트 목표가가 따라오는가?

[최종 판단]
- 리레이팅 가능성: 높음/중간/낮음
- 현금 대비 매력도: ★★★★★ / ★★★☆☆ / ★☆☆☆☆ 중 선택
- 위험요소 3가지
- 구체적 투자 전략 (진입/대기/회피)"""
                            else:
                                prompt_body = f"""아래는 미국 종목 1개의 빠른 분석 요청입니다.

분석 대상:
{stock_line}

이 종목을 간결하게 평가해주세요:
1. 현재 상태 (추세, 촉매, 수급)
2. 단기 전망 (다음 2주)
3. 실전 대응 (진입/대기/회피)
4. 현금 대비 매력도 (★★★★★ ~ ★☆☆☆☆)
5. 주요 리스크"""
                            st.markdown("**📋 생성된 GPT 프롬프트 (복사해서 GPT에 붙여넣으세요):**")
                            st.code(prompt_body, language="text")
                            st.success("✅ 위 프롬프트를 복사하여 GPT에 붙여넣으면 됩니다!")
                        else:
                            st.info(
                                f"**{lookup_query}** 을(를) 스크리너 결과에서 찾을 수 없습니다. "
                                "'⚡ 점수 계산'으로 On-demand 분석을 먼저 실행하면 GPT 분석에도 활용됩니다."
                            )
                    elif market == "KR" and _is_kr_query(lookup_query):
                        od_kr_key   = f"ondemand_kr_{lookup_query}"
                        od_kr_state = st.session_state.get(od_kr_key)
                        if od_kr_state and od_kr_state.get("found"):
                            od_kr_found = od_kr_state["found"]
                            od_row      = od_kr_found["row"]
                            _render_ticker_deep_dive(od_kr_found, "KR")
                            tk   = od_kr_found["ticker"]
                            nm   = od_kr_found.get("_name") or _safe_row_value(od_row, "name", tk) or tk
                            mkt  = od_kr_found.get("_market", "KOSPI")
                            p    = float(_safe_row_value(od_row, "price", 0.0))
                            r20  = float(_safe_row_value(od_row, "ret_20d", 0.0))
                            rs   = float(_safe_row_value(od_row, "rs_rank_pct", 50.0))
                            brk  = float(_safe_row_value(od_row, "breakout_score", 0.0))
                            cat  = float(_safe_row_value(od_row, "catalyst_score", 50.0))
                            rsk  = float(_safe_row_value(od_row, "top_risk_score", 0.0))
                            sup  = float(_safe_row_value(od_row, "supply_score", float("nan")))
                            rr   = float(_safe_row_value(od_kr_found, "rr_score", 0.0))
                            sup_txt = (
                                f" | 수급점수{sup:.0f}"
                                if math.isfinite(sup) else ""
                            )
                            ev_ctx     = od_kr_found.get("kr_event_context") or {}
                            ev_sc      = ev_ctx.get("event_score")
                            kr_adj_sc  = od_kr_found.get("event_adjusted_score")
                            kr_delta   = od_kr_found.get("event_delta", 0.0) or 0.0
                            ev_txt     = f" | DART이벤트{ev_sc:.0f}" if ev_sc is not None else ""
                            kr_adj_txt = (
                                f" | 이벤트조정점수{kr_adj_sc:.0f}(delta{kr_delta:+.0f})"
                                if kr_adj_sc is not None else ""
                            )
                            stock_line = (
                                f"{tk} | {str(nm)[:18]} | 현재가 {p:,.0f}원 | "
                                f"20일{r20:+.1f}% | RS상위{rs:.0f}% | Breakout{brk:.0f} | "
                                f"Catalyst{cat:.0f} | Risk{rsk:.0f} | EPS? | "
                                f"재무? | 리레이팅점수{rr:.0f}{sup_txt}{ev_txt}{kr_adj_txt} [{mkt} On-demand]"
                            )
                            if "리레이팅" in mode:
                                prompt_body = f"""아래는 한국 종목의 리레이팅 가능성 판단 요청입니다.

분석 대상:
{stock_line}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 목적]
이 종목이 한국판 리레이팅 사이클(SK하이닉스, 한미반도체식 급등)에 해당하는가?

아래 5가지를 간결하게 분석해줘:

[Q1] 최근 수익 추정치·목표주가가 올라가는 중인가?
[Q2] EPS 또는 영업이익 성장률이 주가 상승을 정당화하는가?
[Q3] 테마·사이클의 지속성은?
[Q4] 외국인·기관이 동시에 순매수하는가?
[Q5] 코스피/코스닥 대비 상대강도가 개선 중인가?

[최종 판단]
- 리레이팅 가능성: 높음/중간/낮음
- 현금 대비 매력도: ★★★★★ / ★★★☆☆ / ★☆☆☆☆ 중 선택
- 위험요소 3가지
- 구체적 투자 전략 (진입/대기/회피)"""
                            else:
                                prompt_body = f"""아래는 한국 종목 1개의 빠른 분석 요청입니다.

분석 대상:
{stock_line}

이 종목을 간결하게 평가해주세요:
1. 현재 상태 (추세, 촉매, 외국인/기관 수급)
2. 단기 전망 (다음 2주)
3. 실전 대응 (진입/대기/회피)
4. 현금 대비 매력도 (★★★★★ ~ ★☆☆☆☆)
5. 주요 리스크"""
                            st.markdown("**📋 생성된 GPT 프롬프트 (복사해서 GPT에 붙여넣으세요):**")
                            st.code(prompt_body, language="text")
                            st.success("✅ 위 프롬프트를 복사하여 GPT에 붙여넣으면 됩니다!")
                        else:
                            st.info(
                                f"**{lookup_query}** 을(를) 스크리너 결과에서 찾을 수 없습니다. "
                                "'⚡ 점수 계산'으로 KR On-demand 분석을 먼저 실행하면 GPT 분석에도 활용됩니다."
                            )
                    else:
                        st.warning(
                            f"**{lookup_query}** 을(를) 현재 로드된 {market} 결과에서 찾을 수 없습니다. "
                            "기존 스크리너 결과는 그대로 유지됩니다."
                        )
            except Exception as e:
                st.error(f"GPT 프롬프트 생성 오류: {e}")
    
    st.markdown("---")
    all_cands = build_candidates(market, "전체")
    
    # ─ 한국 시장: SK하이닉스, 삼성전자 강제 포함
    if market == "KR":
        from data.kr_universe import KR_FORCED_INCLUDE
        forced_tickers = set(KR_FORCED_INCLUDE.keys())
        existing_tickers = {c["ticker"] for c in all_cands}
        
        # 이미 있는 종목 제외
        missing_tickers = forced_tickers - existing_tickers
        
        # 빠진 강제 포함 종목을 위해 데이터 로드
        if missing_tickers:
            kr_dfs = [
                st.session_state.get("kr_turnaround"),
                st.session_state.get("kr_leader"),
                st.session_state.get("kr_breakout"),
                st.session_state.get("kr_buyable"),
            ]
            fund_map = st.session_state.get("kr_fund", {})
            
            for ticker in missing_tickers:
                for df in kr_dfs:
                    if df is not None and not isinstance(df, tuple) and not df.empty and ticker in df.index:
                        row = df.loc[ticker]
                        fd = fund_map.get(ticker, {})
                        row_dict = dict(row)
                        rr_score = calc_rerating_score(row_dict, fd, {})
                        all_cands.append({
                            "ticker": ticker,
                            "row": row,
                            "track": str(row.get("track", "") or ""),
                            "rr_score": rr_score,
                            "exp": estimate_return(row_dict),
                            "fund": fd,
                            "analyst": {},
                            "upside": float("nan"),
                        })
                        break
    
    # ─ 보유 종목 강제 포함 (시장 무관) — 내 포트폴리오 탭에서 입력한 종목
    #   기준 미달이어도 무조건 포함. 단, 데이터(가격) 수집이 된 종목만 점수화 가능.
    held_tickers = portfolio.get_holding_tickers()
    if held_tickers:
        existing = {c["ticker"].upper() for c in all_cands}
        held_added = []
        held_missing = []
        for tk in held_tickers:
            tk_u = str(tk).strip().upper()
            if tk_u in existing:
                held_added.append(tk_u)   # 이미 후보에 있음 (포함됨)
                continue
            found = _find_ticker_candidate(tk_u, market)
            if found:
                found["_held"] = True
                all_cands.insert(0, found)  # 보유 종목은 맨 앞에
                held_added.append(tk_u)
            else:
                held_missing.append(tk_u)   # 데이터 없어 점수화 불가
        if held_added:
            st.caption(f"💼 보유 종목 {len(held_added)}개 후보 포함: {', '.join(held_added)}")
        if held_missing:
            st.warning(
                f"⚠️ 보유 종목 {len(held_missing)}개는 데이터가 없어 자동 포함되지 않았습니다: "
                f"{', '.join(held_missing)}\n\n"
                f"(이 시장 유니버스에 없거나 가격 수집이 안 된 종목입니다. "
                f"GPT 프롬프트에는 수동으로 추가하거나, 티커가 맞는지 확인하세요.)"
            )
        # 현금 비중도 안내 (비중을 실제 입력한 경우만)
        _alloc = portfolio.compute_allocation(portfolio.get_holdings())
        if _alloc["invested"] > 0:
            st.caption(f"💵 현재 현금 비중 약 {_alloc['cash_pct']:.0f}% "
                       f"(종목 투자 {_alloc['invested']*100:.0f}%) — GPT 현금 순위 평가에 참고하세요")
    
    n_filtered = len(all_cands)
    if n_filtered == 0:
        st.info("스크리너를 먼저 실행해주세요."); return
    _mk = f"gpt_max_{market}"
    # 재스캔으로 후보 수가 바뀌어 이전 세션값이 범위를 벗어나면 초기화
    if _mk in st.session_state and not (6 <= st.session_state[_mk] <= max(6, n_filtered)):
        del st.session_state[_mk]
    if n_filtered <= 6:
        max_total = n_filtered
        st.caption(f"걸러낸 후보 {n_filtered}개 전체 사용")
    else:
        max_total = st.slider("최대 후보 수", 6, n_filtered, n_filtered, 1, key=_mk,
                              help=f"걸러낸 후보 {n_filtered}개가 기본값. 줄이고 싶으면 조절.")
    per_part  = st.slider("프롬프트 당 종목 수", 4, 10, 6, 1, key=f"gpt_per_{market}")

    candidates = all_cands[:max_total]
    if not candidates:
        st.info("스크리너를 먼저 실행해주세요."); return

    # 모드별 정렬 및 필터
    if "급등" in mode:
        # 급등 후보: 백테스트에서 급등(hit15)을 예측한 팩터(돌파+과열+주도) 기준 정렬
        def _spike_score(c):
            r = c["row"]
            return (float(r.get("breakout_score",0) or 0)*0.4
                    + float(r.get("top_risk_score",0) or 0)*0.3
                    + float(r.get("leader_score",0) or 0)*0.3)
        candidates.sort(key=_spike_score, reverse=True)
        mode_label = "단기 급등 후보 (변동성 큰 순)"
    elif "리레이팅" in mode:
        candidates.sort(key=lambda x: x["rr_score"], reverse=True)
        mode_label = "MU식 리레이팅 가능성"
    elif "저평가" in mode:
        def _val_score(c):
            an = c["analyst"]
            up = float(an.get("upside_pct", float("nan")) or float("nan"))
            return up if math.isfinite(up) else -999
        candidates.sort(key=_val_score, reverse=True)
        mode_label = "저평가 (적정가 대비 업사이드)"
    else:
        def _bal_score(c):
            rr = c["rr_score"]
            an = c["analyst"]
            up = float(an.get("upside_pct", float("nan")) or float("nan"))
            up_s = up if math.isfinite(up) else 0
            return rr * 0.5 + up_s * 0.5
        candidates.sort(key=_bal_score, reverse=True)
        mode_label = "균형형 (리레이팅 + 저평가)"

    mkt_name = "미국" if market=="US" else "한국"

    # 후보 목록 표시
    st.markdown(f'<div style="font-size:14px;font-weight:900;color:#f1f5f9;margin-bottom:10px">📋 {mode_label} 후보 {len(candidates)}개</div>', unsafe_allow_html=True)
    for i,c in enumerate(candidates,1):
        rr   = c["rr_score"]
        tk   = c["ticker"]
        row  = c["row"]
        an   = c["analyst"]
        name = row.get("company",row.get("name",tk))
        up   = float(c.get("upside", float("nan")) or float("nan"))
        rr_c = "#4ade80" if rr>=70 else "#fbbf24" if rr>=50 else "#64748b"
        up_c = "#4ade80" if (math.isfinite(up) and up>=15) else "#fbbf24" if (math.isfinite(up) and up>=0) else "#f87171"
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid rgba(255,255,255,0.06);border-radius:7px;'
            f'padding:8px 12px;margin-bottom:5px;display:flex;justify-content:space-between;align-items:center">'
            f'<div style="display:flex;gap:10px;align-items:center">'
            f'<span style="font-size:12px;color:#475569">#{i}</span>'
            f'<span style="font-size:14px;font-weight:800;color:#f1f5f9">{tk}</span>'
            f'<span style="font-size:12px;color:#475569">{str(name)[:18]}</span>'
            f'{_track_badge(c["track"])}</div>'
            f'<div style="display:flex;gap:14px">'
            f'<div style="text-align:center"><div title="rr_score: UI 전용 리레이팅/내러티브 점수입니다." style="font-size:10px;color:#475569">리레이팅(rr)</div><div style="font-size:14px;font-weight:700;color:{rr_c}">{rr:.0f}</div></div>'
            f'<div style="text-align:center"><div style="font-size:10px;color:#475569">업사이드</div><div style="font-size:14px;font-weight:700;color:{up_c}">{f"{up:+.0f}%" if math.isfinite(up) else "—"}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # 프롬프트 생성
    st.markdown('<br>', unsafe_allow_html=True)
    parts = [candidates[i:i+per_part] for i in range(0,len(candidates),per_part)]
    total_parts = len(parts)
    
    # 분석 종목 정보 표시
    st.info(f"📊 **분석 준비**\n\n"
            f"총 분석 대상 종목: **{len(candidates)}개**\n\n"
            f"프롬프트 Part: **{total_parts}개** (1개 Part 당 ~{per_part}개씩)"
    )

    for pi, part in enumerate(parts,1):
        stock_lines = []
        for c in part:
            tk  = c["ticker"]
            row = c["row"]
            an  = c["analyst"]
            fd  = c["fund"]
            nm  = row.get("company",row.get("name",tk))
            p   = float(row.get("price",0) or 0)
            r5  = float(row.get("ret_5d",0) or 0)
            r20 = float(row.get("ret_20d",0) or 0)
            rs  = float(row.get("rs_rank_pct",50) or 50)
            brk = float(row.get("breakout_score",0) or 0)
            cat = float(row.get("catalyst_score",50) or 50)
            rsk = float(row.get("top_risk_score",0) or 0)
            rr  = c["rr_score"]
            eps = an.get("eps_revision","?")
            fg  = fd.get("fund_grade","?")
            cur = "달러" if market=="US" else "원"
            sup_txt = ""
            if market == "KR":
                ssc = row.get("supply_score")
                if isinstance(ssc,(int,float)) and math.isfinite(ssc):
                    sup_txt += f" | 수급점수{ssc:.0f}"
                f5 = row.get("foreign_5d")
                if isinstance(f5,(int,float)) and math.isfinite(f5):
                    sup_txt += f" | 외국인5일{'순매수' if f5>0 else '순매도' if f5<0 else '중립'}"
            stock_lines.append(
                f"{tk} | {str(nm)[:18]} | {c['track'].split('/')[0].strip()} | "
                f"현재가 {p:,.0f}{cur} | 5일{r5:+.1f}% | 20일{r20:+.1f}% | "
                f"RS상위{rs:.0f}% | Breakout{brk:.0f} | Catalyst{cat:.0f} | Risk{rsk:.0f} | "
                f"EPS{eps} | 재무{fg} | 리레이팅점수{rr:.0f}{sup_txt}"
            )
        stock_block = "\n".join(stock_lines)

        if "급등" in mode:
            prompt_body = f"""아래는 내 스크리너가 추린 {mkt_name} 단기 급등 후보야. Part {pi}/{total_parts}.

분석 대상:
{stock_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[중요 — 이 목록의 성격]
이 스크리너는 백테스트 결과, "20일 수익률을 예측하는" 능력은 없고
"단기에 크게 움직일(±15% 급등락) 종목을 골라내는" 능력만 약하게 있다.
즉 이건 '오를 종목'이 아니라 '변동성이 터질 후보'다.
방향(상승/하락)은 점수가 못 맞히므로, 네가 정성적으로 판단해줘.
리레이팅점수는 참고만 하고 맹신하지 마라.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

각 종목에 대해 아래를 짧고 구체적으로 판단해라(추측 금지, 모르면 모른다고).

[1] 지금 움직이는 '실제 촉매'가 있나?
- 최근 뉴스/공시/실적/수주/테마 중 단기 주가를 움직일 구체적 사건
- 촉매가 없으면 "촉매 불명 — 순수 수급/모멘텀성"이라고 명시

[2] 수급이 받쳐주나? (한국은 이게 핵심)
- 제공된 외국인5일 방향·수급점수를 해석하고, 최근 외국인·기관 흐름을 보강 확인
- 수급이 들어오는 초입인가, 이미 많이 들어와 과열인가

[3] 급등이 '초입'인가 '끝물'인가?
- 5일/20일 상승률과 Risk 점수를 보고, 추가 여력 vs 되돌림 위험 판단
- 한국 시장은 평균회귀가 강함 — 이미 급등한 종목의 단기 되돌림 위험을 특히 경계

[4] 빠질 신호 / 함정
- 작전성·테마 소멸·차익실현 구간·실적 노이즈 등 무너질 트리거
- 거래정지·관리종목·유증 등 리스크 점검

[5] 실전 대응 (단타 관점)
- 진입 / 관망 / 회피 중 하나로 명확히
- 진입이면: 트리거 조건, 손절 기준(%), 1차 목표, 예상 보유기간(며칠)
- 변동성이 큰 후보이므로 손절선을 반드시 제시

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[최종 출력]
1. 종목별 1~5 요약 (각 3~4줄)
2. 표: 티커 | 촉매유무 | 수급 | 위치(초입/끝물) | 주요리스크 | 대응(진입/관망/회피) | 손절선
3. 오늘 가장 대응할 만한 1~3개 (굵게) + 그 이유
4. 절대 만지지 말 것 1~3개 + 이유

⚠️ 모든 판단은 단기(며칠~2주) 관점이다. 장기 보유 추천이 아니다."""

        elif "리레이팅" in mode:
            prompt_body = f"""아래는 내 모멘텀 스크리너가 선별한 {mkt_name} 리레이팅 후보야. Part {pi}/{total_parts}.
스크리너 데이터 (차트/거래량/RS/촉매 기반) 가 포함돼 있으니 분석에 활용해줘.

분석 대상:
{stock_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 목적]
"MU처럼 이미 많이 올랐지만, 실적 추정치·산업 병목·테마 때문에 계속 리레이팅될 종목"을 찾는다.
싸서 사는 게 아니라, 비싸도 계속 비싸질 수 있는지 확인하는 게 목적이다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

각 종목에 대해 아래 5가지 질문에 답해라.

[Q1] 실적 이후 컨센서스가 얼마나 올라갔나?
- 최근 실적 발표일, EPS/매출 beat/miss 여부
- 실적 발표 이후 애널리스트 EPS·매출 추정치 변화 (상향/하향/유지, 가능하면 %)
- 다음 분기/연간 가이던스 변화
- 판단: 진짜 리레이팅 진행 중인가, 아니면 단순 모멘텀인가?

[Q2] 다음 12개월 EPS·매출 전망이 주가 상승률을 따라잡나?
- 현재 주가가 이미 많이 올랐다면, 향후 EPS 성장률이 주가 상승을 정당화하는가?
- Forward PER/EV·EBITDA가 역사적 범위에서 어디에 있나?
- 주가가 선반영된 상태인가, 아직 EPS 상향 여지가 있나?

[Q3] TAM·사이클이 1분기짜리인가, 1~2년짜리인가?
- 이 종목이 올라가는 이유 (AI 투자 확대, HBM 수요, 사이버보안 의무화 등) 는 언제까지 지속될 것 같나?
- 이 업황/테마가 정점에 가까운가, 아직 초반인가?
- 가장 비슷한 과거 사이클이 있다면 (반도체 2021, 클라우드 2020 등) 어느 시점에 해당하나?

[Q4] 동종업계도 같이 오르나?
- 경쟁사·동종업계 종목의 최근 퍼포먼스를 비교하라.
- 이 종목만 혼자 오르고 있나, 아니면 섹터 전체가 같이 오르나?
- 혼자만 오른다면 개별 이슈인지, 작전성인지, 진짜 내러티브인지 판단하라.

[Q5] 기관 목표가가 뒤늦게 따라오고 있나?
- 최근 1~3개월 내 애널리스트 커버리지 변화 (신규 커버 시작, 목표가 상향 등)
- 기관/대형 헤지펀드의 포지션 변화가 있다면 언급하라.
- 목표가가 현재가를 이미 앞서는가, 뒤처지는가?
  - 목표가가 현재가보다 뒤처진다면 → 아직 재평가 초기일 수 있음
  - 목표가가 이미 훨씬 올라와 있다면 → 선반영 주의

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[최종 출력 형식]

1. 종목별 5개 질문 요약 (각 3~5줄)

2. 두 순위를 반드시 분리:

=== 순위 A: MU식 리레이팅 가능성 ===
(추정치 상향 + 사이클 지속성 + 기관 재평가 진행 여부 기준)
비싸도 더 비싸질 수 있는 순서

=== 순위 B: 현재가 대비 저평가 ===
(fair_value / current_price - 1 기준)
지금 당장 싼 순서
→ 순위 A와 다를 수 있음을 명시하라

3. 최종 결론
- MU처럼 갈 가능성 1~3순위 (굵게)
- "싸서 좋은 종목" vs "비싸도 더 갈 종목" 절대 섞지 마라
- 리레이팅이 이미 끝난 것 같은 종목 경고
- 지금 사기엔 너무 늦은 종목 vs 아직 초기인 종목 구분

4. 표 요약
티커 | Q1컨센서스상향 | Q2EPS정당화 | Q3사이클단계 | Q4섹터동반 | Q5기관재평가 | 리레이팅등급 | 저평가등급

5. ⭐ 현금(CASH) 순위 평가
위의 모든 분석을 토대로, 지금 현재의 {mkt_name} 시장 상황을 반영하여 다음을 판단하라:
- 주요 이슈: 최근 3개월 시장 흐름 (금리 변화, 수급 쏠림, 지정학적 리스크, 테마 회전 등)
- 각 종목이 그 이슈에서 얼마나 주도하고 있는가?
- 현금(CASH/단기채)이 더 나은 시점인지, 아니면 종목 진입이 나은 시점인지?

=== 순위 C: 현재 국면에서 현금 대비 매력도 ===
(현금이 가장 안전한 선택인 상황 → 순위 최하단에 CASH 추천)
(종목 진입이 매력적인 상황 → 순위 상단부터 적극 추천)

각 종목 우측에 [현금대비 매력도: ★★★★★ / ★☆☆☆☆] 로 표기하라.

예시:
- ★★★★★ = 현금보다 훨씬 나음 (리레이팅+저평가+수급 호황)
- ★★★☆☆ = 현금과 비슷하거나 약간 나음 (리스크 있지만 기회 있음)
- ★☆☆☆☆ = 현금이 훨씬 낫다 (리스크 > 기회)

[현금 추천 판단 시 고려 사항]
- 목표가-현재가 gap 이 좁으면 → 현금 추천
- 동종업계 밸류에이션이 일제히 제상향되고 있으면 → 현금 추천  
- 최근 큰 테마(예: SpaceX상장→우주산업 쏠림)의 수급 역풍을 받고 있으면 → 현금 추천
- 금리 상승기에 고밸류 종목이면 → 현금 추천
- 반대로, 컨센서스 상향이 가중되고, 테마 초입이며, 수급이 들어오면 → 종목 추천"""

        elif "저평가" in mode:
            prompt_body = f"""아래는 {mkt_name} 주식 후보야. Part {pi}/{total_parts}.

분석 대상:
{stock_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 목적]
리레이팅 모멘텀과 무관하게, "지금 가격이 진짜 싼가?"를 판단한다.
MU식 급등 가능성은 여기서 고려하지 않는다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

각 종목에 대해:

1. 밸류에이션 분석
- PER / Forward PER / EV·EBITDA / PSR / FCF Yield 중 해당 기업에 맞는 지표
- 역사적 평균 대비 현재 위치 (싼지 비싼지)
- 동종업계 평균 대비 비교

2. 적정가 산출 (3가지)
- 보수적 시나리오 적정가
- 기본 시나리오 적정가
- 공격적 시나리오 적정가
- 현재가 대비 업사이드/다운사이드 %

3. 저평가 이유와 리스크
- 왜 시장이 지금 이 가격에 주고 있나?
- 저평가가 맞다면 언제 시장이 재평가할 것 같나?
- 함정 요소 (value trap 가능성)

결과를 표로:
티커 | 현재가 | 보수적적정가 | 기본적정가 | 공격적적정가 | 업사이드% | PER비교 | 저평가이유 | 재평가예상시점 | 리스크

⚠️ 주의: 이 분석은 저평가 순위이며, MU식 리레이팅 순위와 다를 수 있다.
저평가 1등이 리레이팅 1등이 아닐 수 있음을 반드시 명시하라."""

        else:
            prompt_body = f"""아래는 {mkt_name} 주식 후보야. Part {pi}/{total_parts}.

분석 대상:
{stock_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 목적]
저평가와 리레이팅 가능성을 동시에 보되, 반드시 두 기준을 분리한다.
"싸고 리레이팅도 가능한 종목"을 최우선으로 찾는다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

각 종목에 대해 아래 두 축을 모두 평가하라.

[축 1 — 저평가 여부]
- 현재 밸류에이션이 역사적/동종업계 대비 싼가?
- 기본 시나리오 적정가 대비 업사이드 %

[축 2 — 리레이팅 가능성]
- 최근 실적 이후 컨센서스 상향 여부
- 다음 12개월 EPS 성장이 현재 주가를 정당화하나?
- 업황/테마 사이클이 1분기짜리인가, 1~2년짜리인가?
- 기관 목표가가 뒤따라오고 있나?

반드시 세 그룹으로 분류하라:

그룹 A: 싸고 리레이팅도 가능 ← 최우선 후보
그룹 B: 비싸지만 리레이팅 가능 ← MU식 후보 (비싸도 더 갈 수 있음)
그룹 C: 싸지만 리레이팅 어려움 ← 가치주 (급등보단 정상화 랠리)

최종 표:
티커 | 적정가업사이드% | 리레이팅등급 | 컨센상향 | 사이클단계 | 기관재평가 | 분류(A/B/C) | 결론"""

        label = f"📋 Part {pi}/{total_parts} ({', '.join(c['ticker'] for c in part)})"
        with st.expander(label, expanded=(pi==1)):
            st.text_area("복사 → ChatGPT/Claude", prompt_body, height=350,
                         key=f"gpt_prompt_{market}_{pi}_{mode[:3]}")

    # 티커만 복사
    all_t = ", ".join(c["ticker"] for c in candidates)
    with st.expander("📌 티커만", expanded=False):
        st.text_area("", all_t, height=55, key=f"gpt_tickers_{market}")

    # ── 🧾 기록용: GPT 통합 JSON 요청 ──────────────────────────────
    st.markdown("---")
    _today = get_data_date() or date.today().isoformat()
    json_prompt = (
        "위에서 네가 분석한 모든 종목을 합쳐, 매수 추천 순위(rank=1이 최우선)로 정렬한 뒤\n"
        "아래 JSON 형식으로만 출력해라. 설명·표 없이 JSON 코드블록 하나만.\n\n"
        "{\n"
        f'  "as_of": "{_today}",\n'
        f'  "market": "{market}",\n'
        '  "picks": [\n'
        '    {"rank":1,"ticker":"089970","name":"브이엠","action":"진입","trigger":"70400 돌파",'
        '"stop_price":65000,"stop_pct":-6,"target_price":85000,"target_pct":18,"note":"1Q 서프라이즈, 과열 낮음"}\n'
        "  ]\n"
        "}\n\n"
        "규칙:\n"
        '- action 은 "진입"/"관망"/"회피" 중 하나.\n'
        "- ticker 는 6자리 숫자코드 문자열.\n"
        "- stop_price 는 숫자(원), 없으면 null. stop_pct 는 음수(예 -6).\n"
        "- target_price 는 목표가(원), target_pct 는 목표 상승률(예 18). 모르면 null.\n"
        '- note 는 한 줄 핵심 판단(촉매/수급/리스크 요약).\n'
        "- 분석한 모든 종목을 빠짐없이 포함. 순위는 매수 우선순위."
    )
    with st.expander("🧾 마지막 단계: 통합 JSON 요청 (기록용)", expanded=False):
        st.caption("Part별 분석을 다 받은 뒤, 이걸 GPT에 마지막 메시지로 보내면 고정 형식이 나옵니다.")
        st.text_area("복사 → GPT 마지막 메시지", json_prompt, height=240,
                     key=f"gpt_json_prompt_{market}")

    # ── 📋 자주 쓰는 후속 프롬프트 (JSON 이후 단계별 질문) ──────────
    with st.expander("📋 자주 쓰는 후속 프롬프트 (복사용)", expanded=False):
        st.caption("JSON을 받은 뒤 GPT에게 이어서 던지는 질문들입니다. "
                   "각 칸을 복사해서 GPT에 순서대로 보내세요. "
                   "보유 종목이 있는 프롬프트는 '내 포트폴리오' 탭 내용이 자동으로 채워집니다.")
        _held = portfolio.get_holdings()
        _alloc = portfolio.compute_allocation(_held)
        for _i, (_title, _body) in enumerate(followup_prompts.FOLLOWUP_PROMPTS):
            _filled = followup_prompts.render_prompt(_body, _held, _alloc)
            st.markdown(f"**{_title}**")
            st.text_area(_title, _filled, height=120,
                         key=f"followup_{market}_{_i}", label_visibility="collapsed")

    # ── 📒 기록 & 검증 ───────────────────────────────────────────
    st.markdown("#### 📒 기록 & 검증")
    paste = st.text_area("GPT가 준 통합 JSON 붙여넣기", height=130, key=f"journal_paste_{market}",
                         placeholder='{"as_of":"2026-06-04","market":"KR","picks":[...]}')
    if st.button("💾 기록 저장", key=f"journal_save_{market}"):
        try:
            parsed = journal.parse_gpt_json(paste)
            d = parsed.get("as_of") or _today
            ep = journal.capture_entry_prices(parsed["picks"], d)
            journal.save(market, d, dict(parsed, entry_prices=ep))
            st.success(f"✅ {d} 기록 저장 — {len(parsed['picks'])}종목 "
                       f"(진입가 {len(ep)}개 박제). 저장 위치: {journal.JOURNAL_DIR}")
        except Exception as e:
            st.error(f"저장 실패: {e}  (JSON 형식을 확인하세요)")

    dates = journal.list_dates(market)
    if dates:
        cda, cdb = st.columns([2, 1])
        with cda:
            sel = st.selectbox("기록 날짜", dates, key=f"journal_date_{market}")
        with cdb:
            eval_days = st.number_input("평가 기간(거래일)", 1, 60, 5, 1,
                                        key=f"journal_eval_{market}")
        entry = journal.load(market, sel)
        if entry:
            rows = journal.score_entry(entry, eval_days=int(eval_days))
            summ = journal.summary(rows)
            if summ["n_entry"]:
                st.caption(
                    f"진입 {summ['n_entry']}종목 · 성공 {summ['entry_win']} / 손절·부진 {summ['entry_loss']} "
                    f"· 승률 {summ['entry_win_rate']}% · 평균수익 {summ['entry_avg_ret']}%  "
                    f"(평가일 기준일+{int(eval_days)}거래일, 데이터 있는 것만)")
            disp = pd.DataFrame([{
                "실제매수": portfolio.is_bought(market, sel, r["ticker"]),
                "순위": r.get("rank"), "티커": r["ticker"], "종목": str(r["name"])[:12],
                "대응": r["action"], "진입가": r.get("entry_close"),
                "목표가": r.get("target_price"),
                "목표업사이드%": r.get("target_up"),
                "평가가": r.get("eval_close"), "수익%": r.get("ret"),
                "목표달성": "🎯" if r.get("target_hit") else "",
                "손절": "Y" if r.get("stop_hit") else "",
                "판정": r.get("verdict"),
                "현재가": r.get("now_close"), "현재수익%": r.get("now_ret"),
                "매도신호": r.get("now_signal"),
                "핵심판단": str(r.get("note") or "")[:40],
                "상태": r.get("status"),
            } for r in rows])
            st.caption("✅ '실제매수' 칸을 체크하면 내가 실제로 산 종목으로 기록됩니다 (포트폴리오 탭에서 한눈에 확인).")
            edited_j = st.data_editor(
                disp, use_container_width=True, hide_index=True,
                key=f"journal_editor_{market}_{sel}",
                disabled=[c for c in disp.columns if c != "실제매수"],
                column_config={
                    "실제매수": st.column_config.CheckboxColumn("실제매수", help="내가 실제로 매수한 종목이면 체크", width="small"),
                },
            )
            if st.button("💾 실제매수 체크 저장", key=f"journal_buy_save_{market}_{sel}"):
                cnt = 0
                for _, er in edited_j.iterrows():
                    tk = er["티커"]
                    chk = bool(er["실제매수"])
                    if chk != portfolio.is_bought(market, sel, tk):
                        portfolio.set_bought(market, sel, tk, bought=chk,
                                             real_price=er.get("진입가"))
                        cnt += 1
                st.success(f"✅ 실제매수 체크 {cnt}건 반영됨"); st.rerun()
            if st.button("🗑 이 날짜 기록 삭제", key=f"journal_del_{market}"):
                journal.delete(market, sel); st.rerun()
    else:
        st.caption("아직 저장된 기록이 없습니다. 위 통합 JSON을 받아 붙여넣고 저장하세요.")


# ════════════════════════════════════════════════════════════════════
# 탭: 💼 내 포트폴리오 (보유 종목 + 실제 매수 추적)
# ════════════════════════════════════════════════════════════════════

def render_portfolio_tab():
    st.markdown("#### 💼 내 보유 종목")
    st.caption(
        "여기 입력한 종목은 **스크리너 선별에 무조건 포함**되고 GPT 프롬프트에도 자동으로 들어갑니다. "
        "저장 위치는 앱 바깥(`~/.etf-radar-cache/portfolio.json`)이라 WSL을 껐다 켜도 유지됩니다."
    )

    # ── 🔎 종목 검색해서 추가 (티커 오타 방지) ──────────────────────
    st.markdown("##### 🔎 종목 검색해서 추가")
    st.caption("종목명 일부(예: '삼성')나 티커(예: NVDA)를 입력하면 후보가 뜹니다. 골라서 추가하면 티커가 정확히 등록됩니다. "
               "※ 미국 종목은 티커로 검색하세요. (회사명 검색은 스크리너를 한 번 돌린 뒤 가능)")

    # 종목명 매핑: 정적(한국+미국) + 세션 수집분(full_df) 합치기
    name_map = portfolio.build_name_map_all()
    for _key in ("kr_full_df", "us_full_df"):
        _df = st.session_state.get(_key)
        if _df is not None and not isinstance(_df, tuple):
            try:
                if "name" in _df.columns:
                    for _t in _df.index:
                        name_map[str(_t)] = str(_df.loc[_t, "name"] or _t)
                else:
                    for _t in _df.index:
                        name_map.setdefault(str(_t), str(_t))
            except Exception:
                pass

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        q = st.text_input("종목명 또는 티커 검색", key="pf_search_q",
                          placeholder="예: 삼성, 하이닉스, NVDA, 005930")
    results = portfolio.search_stocks(q, name_map, limit=20) if q else []

    if q and not results:
        st.warning(f"'{q}'에 해당하는 종목을 찾지 못했습니다. "
                   f"(아직 스크리너를 안 돌렸으면 검색 범위가 좁습니다. "
                   f"정확한 티커를 안다면 아래 표에 직접 입력해도 됩니다.)")
    elif results:
        opts = [f"{nm} ({tk})" for tk, nm in results]
        with sc1:
            picked = st.selectbox("검색 결과에서 선택", opts, key="pf_search_pick")
        with sc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ 보유 종목에 추가", use_container_width=True, key="pf_add_searched"):
                idx = opts.index(picked)
                tk, nm = results[idx]
                portfolio.add_holding(tk, nm)
                st.success(f"✅ {nm}({tk}) 추가됨 — 아래 표에서 비중·평단을 채우세요")
                st.rerun()

    st.markdown("---")
    st.markdown("##### 📝 보유 종목 표 (비중·평단 입력)")

    holdings = portfolio.get_holdings()
    # 데이터에디터용 DataFrame (빈 행 몇 개 포함해 추가 입력 쉽게)
    rows = []
    for tk, h in holdings.items():
        # 등록 상태: name_map에 있으면 ✅, 없으면 ❓ (오타/미수집 의심)
        recognized = tk in name_map
        auto_name = name_map.get(tk, "")
        rows.append({
            "확인": "✅" if recognized else "❓",
            "티커": tk,
            "종목명": h.get("name", "") or auto_name,
            "비중": h.get("weight_raw", ""),
            "평단가": h.get("avg_price"),
            "메모": h.get("memo", ""),
        })
    # 빈 행 3개 추가
    for _ in range(3):
        rows.append({"확인": "", "티커": "", "종목명": "", "비중": "", "평단가": None, "메모": ""})

    df_edit = pd.DataFrame(rows, columns=["확인", "티커", "종목명", "비중", "평단가", "메모"])

    st.caption("💡 **확인** 칸: ✅=종목 인식됨, ❓=티커를 못 찾음(오타 의심 — 위 검색으로 다시 추가 권장). "
               "**비중**은 분수로 입력: `3/5`, `1/4`. 합이 1 미만이면 나머지는 자동 현금.")

    edited = st.data_editor(
        df_edit,
        use_container_width=True,
        num_rows="dynamic",
        key="portfolio_editor",
        column_config={
            "확인": st.column_config.TextColumn("확인", help="✅ 인식됨 / ❓ 못 찾음", width="small", disabled=True),
            "티커": st.column_config.TextColumn("티커", help="검색으로 추가하면 자동으로 정확히 입력됩니다", width="small"),
            "종목명": st.column_config.TextColumn("종목명", width="medium"),
            "비중": st.column_config.TextColumn("비중", help="분수로 입력: 3/5, 1/4 등", width="small"),
            "평단가": st.column_config.NumberColumn("평단가", help="내 평균 매수가 (선택)", format="%.0f"),
            "메모": st.column_config.TextColumn("메모", width="medium"),
        },
    )

    cpa, cpb = st.columns([1, 3])
    with cpa:
        if st.button("💾 보유 종목 저장", type="primary", use_container_width=True, key="pf_save"):
            new_rows = []
            for _, r in edited.iterrows():
                tk = str(r.get("티커", "") or "").strip()
                if not tk:
                    continue
                new_rows.append({
                    "ticker": tk,
                    "name": r.get("종목명", ""),
                    "weight_raw": r.get("비중", ""),
                    "avg_price": r.get("평단가"),
                    "memo": r.get("메모", ""),
                })
            portfolio.set_holdings_bulk(new_rows)
            st.success(f"✅ 보유 종목 {len(new_rows)}개 저장됨")
            st.rerun()
    with cpb:
        tickers = portfolio.get_holding_tickers()
        if tickers:
            st.caption(f"현재 보유: **{len(tickers)}개** — {', '.join(tickers)}")
        else:
            st.caption("아직 저장된 보유 종목이 없습니다. 위 표에 입력하고 저장하세요.")

    # ─ 비중/현금 자동 계산 결과 ──────────────────────────────────────
    if holdings:
        alloc = portfolio.compute_allocation(holdings)
        st.markdown("##### 📊 비중 배분 (현금 자동 계산)")
        if alloc["over"]:
            st.warning(f"⚠️ 종목 비중 합이 **{alloc['invested']*100:.1f}%**로 100%를 넘습니다. 비중을 다시 확인하세요.")
        # 종목 + 현금을 한 표로
        alloc_rows = []
        for r in alloc["rows"]:
            if r["weight"] > 0:
                alloc_rows.append({
                    "티커": r["ticker"],
                    "종목명": r["name"],
                    "입력": r["weight_raw"],
                    "비중%": r["weight_pct"],
                })
        # 현금 행 추가
        alloc_rows.append({
            "티커": "💵 현금", "종목명": "(자동 계산)",
            "입력": "", "비중%": alloc["cash_pct"],
        })
        adf = pd.DataFrame(alloc_rows)
        st.dataframe(
            adf, use_container_width=True, hide_index=True,
            column_config={
                "비중%": st.column_config.ProgressColumn(
                    "비중%", min_value=0, max_value=100, format="%.1f%%"),
            },
        )
        cca, ccb = st.columns(2)
        cca.metric("종목 투자 비중", f"{alloc['invested']*100:.1f}%")
        ccb.metric("💵 현금 비중", f"{alloc['cash_pct']:.1f}%")

    st.markdown("---")
    st.markdown("#### 🛒 실제 매수 기록")
    st.caption(
        "저널(기록 & 검증)에서 GPT가 추천한 종목 중 **실제로 매수한 것**을 체크할 수 있습니다. "
        "각 시장의 'GPT 리레이팅 분석' 탭 하단 기록 표에서 '실제매수' 칸을 체크하세요."
    )

    # 실제 매수한 종목 요약
    pf_data = portfolio._load()
    bought = pf_data.get("bought", {})
    real_bought = {k: v for k, v in bought.items() if v.get("bought")}
    if real_bought:
        b_rows = []
        for key, rec in real_bought.items():
            parts = key.split(":")
            if len(parts) == 3:
                mkt_, d_, tk_ = parts
                b_rows.append({
                    "시장": mkt_, "날짜": d_, "티커": tk_,
                    "실매수가": rec.get("real_price"),
                    "메모": rec.get("memo", ""),
                })
        if b_rows:
            st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("아직 실제 매수로 체크한 종목이 없습니다.")


# ════════════════════════════════════════════════════════════════════
# 탭 3: ⚙️ 고급/진단 (테마입력, 섹터ETF, 백테스트)
# ════════════════════════════════════════════════════════════════════

def render_advanced_tab(market: str):
    adv_tabs = st.tabs(["🎯 테마 입력","📡 섹터/테마 분석","🔬 백테스트/진단","💼 AI 인사이트","📊 재무"])

    with adv_tabs[0]:
        _render_theme_market(market)

    with adv_tabs[1]:
        _render_sector_analysis(market)

    with adv_tabs[2]:
        _render_backtest(market)

    with adv_tabs[3]:
        _render_ai_insight(market)

    with adv_tabs[4]:
        top = st.session_state.get("us_leader" if market=="US" else "kr_leader")
        fund= st.session_state.get("us_fund" if market=="US" else "kr_fund",{})
        _render_fundamental(top, fund, market)


# ── 테마 입력 ─────────────────────────────────────────────────────

def _render_theme_market(market: str):
    mkt_name   = "미국" if market=="US" else "한국"
    theme_key  = "us_themes"  if market=="US" else "kr_themes"
    ticker_key = "us_theme_tickers" if market=="US" else "kr_theme_tickers"
    ctx_key    = "us_context_factors" if market=="US" else "kr_context_factors"
    avoid_key  = "us_avoid_tickers" if market=="US" else "kr_avoid_tickers"
    ts_key     = "us_theme_scores" if market=="US" else "kr_theme_scores"

    prompt = generate_theme_scan_prompt(market)
    with st.expander(f"📋 GPT에게 {mkt_name} 테마 물어보기 (JSON)", expanded=True):
        st.text_area("복사 → GPT/Claude", prompt, height=260, key=f"tp_{market}")

    response = st.text_area("JSON 답변 붙여넣기", height=200, key=f"tr_{market}",
        placeholder='{"market_context":"...","themes":[{"name":"AI메모리","stage":"mid","tickers":[{"ticker":"MU","role":"leader","weight":1.0}]}]}')

    c1,c2 = st.columns([3,1])
    with c1:
        parse_btn = st.button(f"✅ 파싱 적용", use_container_width=True, type="primary", key=f"tp_btn_{market}")
    with c2:
        if st.button("🗑 초기화", use_container_width=True, key=f"tc_btn_{market}"):
            for k in [theme_key,ticker_key,ctx_key,avoid_key,ts_key]:
                st.session_state[k] = [] if isinstance(st.session_state.get(k,{}), list) else {}
            st.rerun()

    if parse_btn and response.strip():
        themes  = parse_theme_response(response, market)
        tickers = get_all_theme_tickers(themes)
        ctx_f   = get_context_factors(response)
        _,avoid_t = get_avoid_list(response)
        if themes:
            st.session_state[theme_key]  = themes
            st.session_state[ticker_key] = tickers
            st.session_state[ctx_key]    = ctx_f
            st.session_state[avoid_key]  = avoid_t
            st.success(f"✅ {len(themes)}개 테마, {len(tickers)}개 종목 → ↺ 재실행 권장")
        else:
            st.error("JSON 파싱 실패")

    # 한국 테마 그룹 힌트 표시
    if market == "KR":
        try:
            from data.kr_universe import KR_THEME_GROUPS
            with st.expander("📋 한국 테마 그룹 참고 (클릭 확장)", expanded=False):
                for theme_name, info in KR_THEME_GROUPS.items():
                    tickers_str = ", ".join(info["tickers"][:8])
                    st.markdown(
                        f'<div style="background:#0f172a;border-left:3px solid #f59e0b;border-radius:6px;padding:8px 12px;margin-bottom:5px">' +
                        f'<div style="display:flex;justify-content:space-between;align-items:center">' +
                        f'<div><span style="font-size:13px;font-weight:800;color:#f59e0b">{theme_name}</span> ' +
                        f'<span style="font-size:11px;color:#475569">{info["desc"]}</span></div>' +
                        f'<span style="font-size:11px;color:#334155">미국유사: {info["us_analog"]}</span></div>' +
                        f'<div style="font-size:11px;color:#334155;margin-top:3px">{tickers_str}...</div></div>',
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass

    saved = st.session_state.get(theme_key,[])
    saved_ts = st.session_state.get(ts_key,[])
    ts_map = {ts.get("name",""): ts for ts in saved_ts}
    colors = ["#f59e0b","#818cf8","#4ade80","#60a5fa","#f87171"]

    for i,th in enumerate(saved):
        c = colors[i%len(colors)]
        ts = ts_map.get(th.get("name",""),{})
        ts_score = ts.get("theme_score",0)
        pills = "".join(
            f'<span style="background:rgba(255,255,255,0.06);color:#f1f5f9;padding:1px 7px;border-radius:3px;font-size:12px;margin:2px">'
            f'{(t.get("ticker","") if isinstance(t,dict) else t)}</span>'
            for t in th.get("tickers",[])
        )
        ts_c = "#4ade80" if ts_score>=70 else "#fbbf24" if ts_score>=50 else "#64748b"
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid {c}33;border-left:3px solid {c};'
            f'border-radius:8px;padding:10px 14px;margin-bottom:7px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">'
            f'<span style="font-size:14px;font-weight:800;color:{c}">{th["name"]}</span>'
            + (f'<span style="font-size:13px;font-weight:900;color:{ts_c}">TS:{ts_score:.0f}</span>' if ts_score else "")
            + f'</div><div style="flex-wrap:wrap;display:flex;gap:3px">{pills}</div></div>',
            unsafe_allow_html=True,
        )


# ── 섹터 분석 ─────────────────────────────────────────────────────

def _render_sector_analysis(market: str):
    sector_data = st.session_state.get("sector_analysis",[])
    us_t = st.session_state.get("universe_themes_us",[])
    kr_t = st.session_state.get("universe_themes_kr",[])
    themes = us_t if market=="US" else kr_t

    if sector_data:
        st.markdown("#### 📡 섹터 ETF 수익률")
        hot  = [s for s in sector_data if s.get("hot")]
        cold = [s for s in sector_data if not s.get("hot")]
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**🔥 강한 섹터**")
            for s in hot[:7]:
                r20=s.get("ret20",0); r5=s.get("ret5",0)
                st.markdown(
                    f'<div style="background:#0f172a;border-left:3px solid #4ade80;border-radius:6px;padding:7px 11px;margin-bottom:5px;display:flex;justify-content:space-between">'
                    f'<div><span style="font-size:13px;font-weight:700;color:#f1f5f9">{s["sector"]}</span> <span style="font-size:11px;color:#475569">{s["etf"]}</span></div>'
                    f'<div style="display:flex;gap:10px"><span style="color:{"#4ade80" if r5>=0 else "#f87171"};font-size:12px;font-weight:700">{r5:+.1f}%</span>'
                    f'<span style="color:{"#4ade80" if r20>=0 else "#f87171"};font-size:12px;font-weight:700">{r20:+.1f}%</span></div></div>',
                    unsafe_allow_html=True,
                )
        with c2:
            st.markdown("**❄ 약한/소외 섹터**")
            for s in cold[-5:]:
                r20=s.get("ret20",0)
                st.markdown(
                    f'<div style="background:#0f172a;border-left:3px solid #334155;border-radius:6px;padding:7px 11px;margin-bottom:5px;display:flex;justify-content:space-between">'
                    f'<span style="font-size:13px;color:#64748b">{s["sector"]} <span style="font-size:11px">{s["etf"]}</span></span>'
                    f'<span style="color:{"#4ade80" if r20>=0 else "#f87171"};font-size:12px;font-weight:700">{r20:+.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

    if themes:
        st.markdown("#### 🌐 전체 유니버스 테마 강도")
        for th in themes[:8]:
            r20=th.get("avg_ret20",0); hot=th.get("hot",False)
            c="#f59e0b" if hot else "#475569"
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid {c}33;border-left:3px solid {c};border-radius:6px;padding:8px 12px;margin-bottom:5px;display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-size:13px;font-weight:700;color:{c}">{"🔥 " if hot else ""}{th["theme"]}</span>'
                f'<span style="color:{"#4ade80" if r20>=0 else "#f87171"};font-size:13px;font-weight:700">{r20:+.1f}%</span></div>',
                unsafe_allow_html=True,
            )
    if not sector_data and not themes:
        st.info("스크리너 실행 후 자동으로 표시됩니다.")

# ── 백테스트 ─────────────────────────────────────────────────────

def _render_backtest(market: str = "US"):
    # 시장에 맞는 백테스트 결과 선택 (절대 교차 없음)
    if market == "KR":
        bt = st.session_state.get("backtest_kr_result")  # 한국만
        bench_name = "KOSPI"
    else:
        bt = st.session_state.get("backtest_result")      # 미국만
        bench_name = "SPY"
    # 시장 불일치 방지: KR 결과에 미국 데이터가 섞이면 무시
    if bt and market == "KR" and bt.get("market","US") == "US":
        bt = None
    if bt and market == "US" and bt.get("market","US") == "KR":
        bt = None

    analyst = st.session_state.get("us_analyst",{}) if market=="US" else {}
    earnings = get_earnings_calendar(analyst) if analyst else []

    if earnings:
        st.markdown("#### 📅 실적 발표 예정 (4주 이내)")
        for ev in earnings[:8]:
            dte=ev["days_to_earnings"]; rev=ev["eps_revision"]
            rev_c="#4ade80" if rev=="up" else "#f87171" if rev=="down" else "#64748b"
            urg_c="#f87171" if dte<=7 else "#fbbf24" if dte<=14 else "#60a5fa"
            ldr=st.session_state.get("us_leader")
            name=ev["ticker"]
            if ldr is not None and not isinstance(ldr,tuple) and ev["ticker"] in ldr.index:
                name=f"{str(ldr.loc[ev['ticker'],'company'])}({ev['ticker']})"
            st.markdown(
                f'<div style="background:#0f172a;border-left:3px solid {urg_c};border-radius:7px;padding:8px 12px;margin-bottom:5px;display:flex;justify-content:space-between;align-items:center">' +
                f'<div><span style="background:{urg_c}22;color:{urg_c};padding:1px 7px;border-radius:3px;font-size:12px;font-weight:700">D-{dte}</span> ' +
                f'<span style="font-size:13px;font-weight:700;color:#f1f5f9">{name}</span> ' +
                f'<span style="font-size:12px;font-weight:700;color:{rev_c}">{"▲EPS상향" if rev=="up" else "▼하향" if rev=="down" else "중립"}</span></div>' +
                f'<span style="font-size:12px;color:#475569">{ev.get("next_earnings","")}</span></div>',
                unsafe_allow_html=True,
            )

    if not bt:
        mkt_label = "🇰🇷 한국" if market=="KR" else "🇺🇸 미국"
        st.info(f"{mkt_label} 백테스트 미실행 — {mkt_label} 스크리너 실행 후 자동 계산됩니다.")
        return

    ov  = bt.get("overall",{})
    reg = bt.get("regression",{})

    st.markdown(
        '<div style="background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.2);' +
        'border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:#64748b">' +
        f'⚠ 아래 수익률은 <strong>원수익률(절대)</strong>과 <strong>{bench_name} 초과수익(알파)</strong>로 나뉩니다. ' +
        f'강세장에서는 원수익률이 높아도 {bench_name} 초과가 낮을 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

    spm = bt.get("spearman", {})
    metrics = [
        ("평균수익률",    f"{ov.get('avg_ret',0):+.1f}%",    "#4ade80" if ov.get("avg_ret",0)>=0 else "#f87171"),
        ("중앙값",       f"{ov.get('median_ret',0):+.1f}%",  "#4ade80" if ov.get("median_ret",0)>=0 else "#f87171"),
        ("승률(원)",     f"{ov.get('win_rate',0):.0f}%",     "#4ade80" if ov.get("win_rate",0)>=55 else "#f87171"),
    ]
    if market == "KR":
        kospi_exc = float(ov.get("avg_excess_kospi", ov.get("avg_excess_spy", float("nan"))) or float("nan"))
        kospi_wr  = float(ov.get("kospi_beat_rate", ov.get("spy_beat_rate", float("nan"))) or float("nan"))
        kosdaq_exc = float(ov.get("avg_excess_kosdaq", float("nan")) or float("nan"))
        kosdaq_wr  = float(ov.get("kosdaq_beat_rate", float("nan")) or float("nan"))
        metrics += [
            ("KOSPI초과",  f"{kospi_exc:+.1f}%" if math.isfinite(kospi_exc) else "—", "#4ade80" if (math.isfinite(kospi_exc) and kospi_exc>=0) else "#f87171"),
            ("KOSPI승률",  f"{kospi_wr:.0f}%"  if math.isfinite(kospi_wr) else "—",  "#4ade80" if (math.isfinite(kospi_wr) and kospi_wr>=50) else "#f87171"),
            ("KOSDAQ초과", f"{kosdaq_exc:+.1f}%" if math.isfinite(kosdaq_exc) else "—", "#4ade80" if (math.isfinite(kosdaq_exc) and kosdaq_exc>=0) else "#f87171"),
            ("순위상관",    f"{spm.get('composite',0):+.3f}", "#4ade80" if spm.get("composite",0)>0.05 else "#64748b"),
            ("R²",          f"{reg.get('r2',0):.3f}", "#4ade80" if reg.get("r2",0)>=0.05 else "#64748b"),
        ]
    else:
        exc_v = float(ov.get("avg_excess_spy") or float("nan"))
        sbr_v = float(ov.get("spy_beat_rate")  or float("nan"))
        metrics += [
            (f"{bench_name}초과", f"{exc_v:+.1f}%" if math.isfinite(exc_v) else "—", "#4ade80" if (math.isfinite(exc_v) and exc_v>=0) else "#f87171"),
            (f"{bench_name}승률", f"{sbr_v:.0f}%"  if math.isfinite(sbr_v) else "—", "#4ade80" if (math.isfinite(sbr_v) and sbr_v>=50) else "#f87171"),
            ("순위상관",    f"{spm.get('composite',0):+.3f}", "#4ade80" if spm.get("composite",0)>0.05 else "#64748b"),
            ("R²",           f"{reg.get('r2',0):.3f}", "#4ade80" if reg.get("r2",0)>=0.05 else "#64748b"),
        ]

    cols = st.columns(len(metrics))
    for col,(label,val,color) in zip(cols,metrics):
        with col:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:10px;text-align:center">' +
                f'<div style="font-size:10px;color:#475569;margin-bottom:3px">{label}</div>' +
                f'<div style="font-size:18px;font-weight:900;color:{color}">{val}</div></div>',
                unsafe_allow_html=True,
            )

    by_track = bt.get("by_track",{})
    if by_track:
        st.markdown("**📊 날짜별 포트폴리오 성과 (각 날짜 상위 K개)**")
        for label,data in sorted(by_track.items()):
            avg=data.get("avg_ret",0); n=data.get("n",0)
            exc=float(data.get("avg_excess_spy") or float("nan"))
            wr=data.get("win_rate",50)
            sbr=float(data.get("spy_beat_rate") or float("nan"))
            avg_c="#4ade80" if avg>=3 else "#fbbf24" if avg>=0 else "#f87171"
            exc_c="#4ade80" if (math.isfinite(exc) and exc>0) else "#f87171"
            extra_html = ""
            if market == "KR":
                kq = float(data.get("avg_excess_kosdaq", float("nan")) or float("nan"))
                kq_wr = float(data.get("kosdaq_beat_rate", float("nan")) or float("nan"))
                kq_c = "#4ade80" if (math.isfinite(kq) and kq > 0) else "#f87171"
                extra_html = (
                    f'<div style="text-align:center"><div style="font-size:10px;color:#475569">KOSDAQ초과</div><div style="font-size:14px;font-weight:700;color:{kq_c}">{f"{kq:+.1f}%" if math.isfinite(kq) else "—"}</div></div>'
                    f'<div style="text-align:center"><div style="font-size:10px;color:#475569">KOSDAQ승률</div><div style="font-size:13px;font-weight:700;color:{"#4ade80" if (math.isfinite(kq_wr) and kq_wr>=50) else "#f87171"}">{f"{kq_wr:.0f}%" if math.isfinite(kq_wr) else "—"}</div></div>'
                )
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid rgba(255,255,255,0.06);border-radius:7px;padding:9px 14px;margin-bottom:5px;display:flex;align-items:center;gap:16px;flex-wrap:wrap">' +
                f'<div style="min-width:80px;font-size:13px;font-weight:700;color:#f1f5f9">{label}</div>' +
                f'<div style="text-align:center"><div style="font-size:10px;color:#475569">원수익률</div><div style="font-size:14px;font-weight:700;color:{avg_c}">{avg:+.1f}%</div></div>' +
                f'<div style="text-align:center"><div style="font-size:10px;color:#475569">{bench_name}초과</div><div style="font-size:14px;font-weight:700;color:{exc_c}">{f"{exc:+.1f}%" if math.isfinite(exc) else "—"}</div></div>' +
                f'<div style="text-align:center"><div style="font-size:10px;color:#475569">승률</div><div style="font-size:13px;font-weight:700;color:{"#4ade80" if wr>=55 else "#f87171"}">{wr:.0f}%</div></div>' +
                f'<div style="text-align:center"><div style="font-size:10px;color:#475569">{bench_name}승률</div><div style="font-size:13px;font-weight:700;color:{"#4ade80" if (math.isfinite(sbr) and sbr>=50) else "#f87171"}">{f"{sbr:.0f}%" if math.isfinite(sbr) else "—"}</div></div>' +
                extra_html +
                f'<div style="font-size:11px;color:#334155">{n}개 날짜</div></div>',
                unsafe_allow_html=True,
            )

    score_buckets = bt.get("score_buckets", [])
    if score_buckets:
        st.markdown("**🎯 점수구간별 hit-rate / 초과수익**")
        try:
            sb_df = pd.DataFrame(score_buckets)
            if market == "KR":
                sb_df = sb_df[[c for c in sb_df.columns if not str(c).upper().startswith("SPY")]]
            st.dataframe(sb_df, use_container_width=True, hide_index=True, height=min(260, 70 + 36 * len(sb_df)))
        except Exception:
            pass

    fstats = bt.get("filter_stats", {})
    if market == "KR" and fstats:
        with st.expander("🧹 KR 백테스트 필터 적용 내역", expanded=False):
            st.markdown(
                '<div style="font-size:12px;color:#475569;margin-bottom:8px">'
                '⚠ 저가주/거래대금 제외는 <strong>날짜×종목 레코드 수</strong>로 카운트됩니다 (종목 수 아님).'
                '</div>', unsafe_allow_html=True)
            labels = {
                "universe_total":       ("초기 데이터",          "종목"),
                "short_history":        ("이력 부족 제외",        "종목"),
                "bad_security":         ("ETF/ETN/SPAC/우선주 제외","종목"),
                "suspicious_daily_move":("수정주가 의심 제외",    "종목"),
                "eligible_tickers":     ("1차 통과 종목",         "종목"),
                "low_price":            ("저가주 제외",           "레코드"),
                "low_liquidity":        ("거래대금 부족 제외",    "레코드"),
                "return_outlier":       ("수익률 이상값 제외",    "레코드"),
                "records_before_filter":("수익률 필터 전",        "레코드"),
                "records_after_filter": ("최종",                  "레코드"),
            }
            fs_rows = [
                {"항목": labels[k][0], "값": v, "단위": labels[k][1]}
                for k, v in fstats.items() if k in labels
            ]
            st.dataframe(pd.DataFrame(fs_rows), use_container_width=True, hide_index=True)

    # 백테스트 감사표
    df_rec = bt.get("df_records")
    if df_rec is not None and not df_rec.empty:
        st.markdown("---")
        st.markdown("#### 🔎 백테스트 감사표")
        st.markdown('<div style="font-size:12px;color:#475569;margin-bottom:10px">어떤 데이터로 예측했고 실제로 어떻게 됐는지 직접 확인</div>', unsafe_allow_html=True)

        fc1,fc2,fc3 = st.columns(3)
        with fc1:
            dates_avail = sorted(df_rec["decision_date"].unique().tolist()) if "decision_date" in df_rec.columns else []
            sel_date = st.selectbox("기준일", ["전체"]+dates_avail[-20:], key=f"bt_audit_date_{market}")
        with fc2:
            top_n_a = st.selectbox("상위 N개", [10,20,50,100,999], key=f"bt_audit_topn_{market}")
        with fc3:
            sort_c = st.selectbox("정렬", ["composite","ret","excess_spy","leader","breakout"], key=f"bt_audit_sort_{market}")

        df_show = df_rec.copy()
        if sel_date != "전체" and "decision_date" in df_show.columns:
            df_show = df_show[df_show["decision_date"]==sel_date]
        sc = sort_c if sort_c in df_show.columns else "composite"
        df_show = df_show.sort_values(sc, ascending=False)
        if sel_date == "전체" and "decision_date" in df_show.columns:
            df_show = df_show.groupby("decision_date").head(top_n_a).reset_index(drop=True)
        else:
            df_show = df_show.head(top_n_a)

        show_cols = [c for c in [
            "decision_date","ticker","name","market","composite","leader","breakout","catalyst","volume","quality","risk",
            "entry_date","entry_price","exit_date","exit_price",
            "ret","spy_ret","excess_spy","kospi_ret","excess_kospi","kosdaq_ret","excess_kosdaq","qqq_ret",
            "predicted_ret","prediction_error",   # 예측 vs 실제 오차
        ] if c in df_show.columns]
        if show_cols:
            st.dataframe(df_show[show_cols].round(2), use_container_width=True, height=380)
            st.markdown(f'<div style="font-size:11px;color:#334155">{len(df_show)}행</div>', unsafe_allow_html=True)

        # 산점도
        if "composite" in df_rec.columns and "ret" in df_rec.columns:
            st.markdown("**📈 합성점수 → 실제수익률 산점도**")
            try:
                import plotly.express as px
                base_cols = ["composite", "ret", "ticker"] + (["name"] if "name" in df_rec.columns else [])
                pdata = df_rec[base_cols].dropna(subset=["composite", "ret", "ticker"]).copy()
                color_col = "excess_kospi" if market == "KR" and "excess_kospi" in df_rec.columns else "excess_spy" if "excess_spy" in df_rec.columns else None
                if color_col:
                    pdata[color_col] = df_rec.loc[pdata.index, color_col]
                pdata = pdata.sample(min(2000,len(pdata)), random_state=42)
                hover_cols = ["ticker"] + (["name"] if "name" in pdata.columns else [])
                fig = px.scatter(pdata, x="composite", y="ret", hover_data=hover_cols,
                    color=color_col,
                    color_continuous_scale="RdYlGn",
                    labels={"composite":"합성점수","ret":"실제20일수익률(%)", color_col or "":"벤치초과(%)"},
                    title=f"합성점수 vs 실제수익률 (R²={reg.get('r2',0):.4f}, Spearman={spm.get('composite',0):+.4f})",
                    template="plotly_dark", opacity=0.5, height=380)
                if reg.get("slope"):
                    xr=[float(pdata["composite"].min()),float(pdata["composite"].max())]
                    yr=[reg["slope"]*x+reg["intercept"] for x in xr]
                    fig.add_scatter(x=xr,y=yr,mode="lines",name="회귀선",line=dict(color="#f59e0b",width=2))
                fig.update_layout(paper_bgcolor="#0f172a",plot_bgcolor="#0f172a",
                                   font_color="#94a3b8",margin=dict(l=40,r=20,t=40,b=40))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("산점도: `pip install plotly` 필요")

    # 스냅샷 백테스트
    snapshots = st.session_state.get("snapshot_results",[])
    if snapshots:
        st.markdown("---")
        st.markdown('<div style="font-size:15px;font-weight:900;color:#f1f5f9;margin-bottom:8px">📸 스냅샷 백테스트</div>', unsafe_allow_html=True)
        for snap in snapshots:
            avg=snap.get("avg_ret",0); wr=snap.get("win_rate",50); vs=snap.get("vs_spy",float("nan"))
            avg_c="#4ade80" if avg>=5 else "#fbbf24" if avg>=0 else "#f87171"
            vs_c="#4ade80" if (math.isfinite(vs) and vs>0) else "#f87171"
            vs_str=f" | vs SPY {vs:+.1f}%" if math.isfinite(vs) else ""
            with st.expander(f"📅 {snap['snapshot_date']} → avg {avg:+.1f}% | 승률 {wr:.0f}%{vs_str}", expanded=False):
                top=snap.get("top_tickers",[])
                if top:
                    import pandas as _pd
                    rows=[{"순위":s["rank"],"티커":s["ticker"],"트랙":s["track_hint"],
                           "점수":s["composite"],"진입가":s["entry_price"],"청산가":s["exit_price"],"수익률%":s["ret"]} for s in top]
                    df_s=_pd.DataFrame(rows)
                    st.dataframe(df_s, use_container_width=True, hide_index=True)
                    st.markdown(
                        f'<div style="display:flex;gap:14px;margin-top:8px">' +
                        f'<div style="text-align:center"><div style="font-size:10px;color:#475569">평균</div><div style="font-size:15px;font-weight:900;color:{avg_c}">{avg:+.1f}%</div></div>' +
                        f'<div style="text-align:center"><div style="font-size:10px;color:#475569">승률</div><div style="font-size:14px;font-weight:700;color:{"#4ade80" if wr>=55 else "#f87171"}">{wr:.0f}%</div></div>' +
                        (f'<div style="text-align:center"><div style="font-size:10px;color:#475569">vs SPY</div><div style="font-size:14px;font-weight:700;color:{vs_c}">{vs:+.1f}%</div></div>' if math.isfinite(vs) else "") +
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    st.markdown('<div style="font-size:11px;color:#334155;margin-top:8px">⚠ 과거 성과가 미래를 보장하지 않습니다. 현재 유니버스 기반 시뮬레이션.</div>', unsafe_allow_html=True)


    # ── Factor 개별 예측력 진단 ───────────────────────────────────────
    df_rec = bt.get("df_records") if bt else None
    if df_rec is not None and not df_rec.empty:
        st.markdown("---")
        with st.expander("🔬 Factor 개별 예측력 진단", expanded=False):
            bench_col = "excess_market" if market == "KR" and "excess_market" in df_rec.columns else                         "excess_kospi" if market == "KR" and "excess_kospi" in df_rec.columns else "excess_spy"

            target_options = {"20일 원수익률": "ret", "시장 초과수익": bench_col}
            if "max_ret_20d" in df_rec.columns:
                target_options["20일 내 최고수익률"] = "max_ret_20d"
            sel_target = st.selectbox("진단 기준", list(target_options.keys()), key=f"fp_target_{market}")
            target_col = target_options[sel_target]

            st.markdown(
                f'<div style="font-size:12px;color:#475569;margin-bottom:10px">'
                f'각 Factor가 <b>{sel_target}</b>과 얼마나 관련 있는지 측정. '
                f'Spearman &gt; 0.05, 상하spread &gt; 2%면 유의미.</div>',
                unsafe_allow_html=True,
            )
            fp = analyze_factor_power(df_rec, bench_col, target_col=target_col)
            if not fp.empty:
                # 판정별 색상
                def _color_verdict(val):
                    c = {"✅ 강":"#4ade80","🟡 보통":"#fbbf24","🔴 역효과":"#f87171","⚪ 약":"#475569"}.get(str(val),"")
                    return f"color:{c}" if c else ""
                try:
                    st.dataframe(
                        fp.style.applymap(_color_verdict, subset=["판정"])
                               .format({c:"{:.4f}" for c in ["Spearman","Pearson","R²"] if c in fp.columns}),
                        use_container_width=True, hide_index=True,
                    )
                except Exception:
                    st.dataframe(fp, use_container_width=True, hide_index=True)

                # 경고
                weak = fp[fp["판정"].isin(["🔴 역효과","⚪ 약"])]["factor"].tolist()
                strong = fp[fp["판정"] == "✅ 강"]["factor"].tolist()
                if weak:
                    st.warning(f"⚠ 예측력 약한 Factor: **{', '.join(weak)}** — 가중치 축소 검토")
                if strong:
                    st.success(f"✅ 예측력 있는 Factor: **{', '.join(strong)}** — 가중치 유지/강화")
                if fp["Spearman"].dropna().abs().max() < 0.05:
                    st.error("🔴 전체 Factor의 Spearman이 0에 가깝습니다. 점수 로직 자체 재검토 필요.")

        # ── hit15 중심 급등 압축력 진단 ─────────────────────────
        with st.expander("🚀 급등 압축력 진단 (hit15 기준) — 상위권에서 +15% 달성률", expanded=False):
            st.markdown(
                '<div style="font-size:12px;color:#475569;margin-bottom:8px">'
                'R²보다 중요: <b>상위 20%에서 hit15 달성률이 전체 평균보다 높은가?</b><br>'
                'Lift > 10%면 이 Factor가 급등주를 실제로 압축하는 데 유효합니다.</div>',
                unsafe_allow_html=True,
            )
            bench_col_hit = ("excess_market" if market=="KR" and "excess_market" in df_rec.columns else
                             "excess_kospi"  if market=="KR" and "excess_kospi"  in df_rec.columns else "excess_spy")
            # 날짜별 정규화 후 진단
            factor_cols_all = ["leader","volume","breakout","catalyst","quality","risk","composite",
                                "momentum_accel","volume_shock","brk_persist","near_high","vol_contract",
                                "pullback_q","risk_filter","volume_enhanced","breakout_enhanced","risk_combined"]
            df_norm = normalize_by_date(df_rec, [c for c in factor_cols_all if c in df_rec.columns])
            hit15_df = analyze_factor_hit15(df_norm, bench_col_hit)
            if not hit15_df.empty:
                def _color_lift(val):
                    try:
                        v = float(val)
                        if v > 10:  return "color:#4ade80;font-weight:bold"
                        elif v > 5: return "color:#fbbf24"
                        elif v < 0: return "color:#f87171"
                    except: pass
                    return ""
                def _color_verdict(val):
                    c={"✅ 강 (급등 압축력 높음)":"#4ade80","🟡 보통":"#fbbf24","🔴 역효과":"#f87171"}.get(str(val),"")
                    return f"color:{c}" if c else ""
                try:
                    st.dataframe(
                        hit15_df.style
                            .applymap(_color_lift,    subset=["Lift(상위-전체)"])
                            .applymap(_color_verdict, subset=["판정"]),
                        use_container_width=True, hide_index=True,
                    )
                except Exception:
                    st.dataframe(hit15_df, use_container_width=True, hide_index=True)

                # 유효 factor 요약
                valid_factors = hit15_df[hit15_df["판정"].str.startswith("✅")]["factor"].tolist()
                weak_factors  = hit15_df[hit15_df["판정"].str.startswith("🔴")]["factor"].tolist()
                if valid_factors:
                    st.success(f"✅ 급등 압축력 있는 Factor: **{', '.join(valid_factors)}**")
                if weak_factors:
                    st.error(f"🔴 역효과 Factor (가중치 0으로 낮출 것): **{', '.join(weak_factors)}**")
                if not valid_factors:
                    st.error("🔴 유효한 Factor 없음 — Factor 로직 재검토 또는 데이터 더 쌓기 필요")
            else:
                st.warning("hit15 데이터 부족 — 백테스트 재실행 필요")

        with st.expander("📊 Factor 5분위 단조성 분석", expanded=False):
            st.markdown(
                '<div style="font-size:12px;color:#475569;margin-bottom:10px">'
                '점수가 높아질수록 수익률이 꾸준히 오르면 유효한 Factor.</div>',
                unsafe_allow_html=True,
            )
            bench_col2 = ("excess_market" if market == "KR" and "excess_market" in df_rec.columns else
                           "excess_kospi"  if market == "KR" and "excess_kospi"  in df_rec.columns else "excess_spy")
            quintiles = analyze_factor_quintiles(df_rec, bench_col2, target_col=target_col)
            sel_factor = st.selectbox("Factor 선택", list(quintiles.keys()), key=f"qt_factor_{market}")
            if sel_factor in quintiles:
                qt_df = pd.DataFrame(quintiles[sel_factor])
                st.dataframe(qt_df, use_container_width=True, hide_index=True)
                # 단조성 판정
                tgt_key = f"평균({target_col})" if f"평균({target_col})" in (quintiles[sel_factor][0] if quintiles[sel_factor] else {}) else "평균수익"
                avgs = [r.get(tgt_key) for r in quintiles[sel_factor] if r.get(tgt_key) is not None]
                if len(avgs) >= 4:
                    is_mono = all(avgs[i] <= avgs[i+1] for i in range(len(avgs)-1))
                    if is_mono:
                        st.success("✅ 단조 증가 — 점수가 높을수록 수익률이 꾸준히 오름")
                    else:
                        st.warning("⚠ 비단조 — 점수 구간별 수익률이 일관되지 않음")

        # ── 가중치 수동 조정 슬라이더 ─────────────────────────────────
        with st.expander("🎚 가중치 수동 조정 & 즉시 성과 확인", expanded=False):
            st.markdown(
                '<div style="font-size:12px;color:#475569;margin-bottom:10px">'
                '슬라이더를 움직이면 해당 가중치로 상위 10개/20개 성과를 즉시 계산합니다.</div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                sl_leader   = st.slider("Leader",   0.10, 0.60, 0.38, 0.01, key=f"sl_lead_{market}")
                sl_volume   = st.slider("Volume",   0.05, 0.40, 0.18, 0.01, key=f"sl_vol_{market}")
            with c2:
                sl_breakout = st.slider("Breakout", 0.05, 0.30, 0.12, 0.01, key=f"sl_brk_{market}")
                sl_catalyst = st.slider("Catalyst", 0.05, 0.30, 0.15, 0.01, key=f"sl_cat_{market}")
            with c3:
                sl_quality  = st.slider("Quality",  0.05, 0.30, 0.10, 0.01, key=f"sl_qot_{market}")
                sl_risk     = st.slider("Risk×",    0.01, 0.20, 0.08, 0.01, key=f"sl_risk_{market}")

            # 합산 정규화
            total = sl_leader + sl_volume + sl_breakout + sl_catalyst + sl_quality
            if total > 0:
                norm = 1.0 / total
                manual_w = {
                    "leader":    round(sl_leader   * norm, 3),
                    "volume":    round(sl_volume   * norm, 3),
                    "breakout":  round(sl_breakout * norm, 3),
                    "catalyst":  round(sl_catalyst * norm, 3),
                    "quality":   round(sl_quality  * norm, 3),
                    "risk_penalty": sl_risk,
                }
                st.markdown(f'<div style="font-size:11px;color:#64748b">정규화 후: {manual_w}</div>', unsafe_allow_html=True)

            bench_col3 = ("excess_market" if market == "KR" and "excess_market" in df_rec.columns else
                          "excess_kospi"  if market == "KR" and "excess_kospi"  in df_rec.columns else "excess_spy")
            obj_options = {
                "20일 원수익률": "ret20",
                "20일 시장초과수익": "excess",
                "20일 내 최고수익률 (급등 탐지)": "max_ret",
                f"상위 N개 중 20거래일 내 +15% 도달 비율": "hit15",
                "상위 10개 기준 +15% 도달 비율 (고정)": "top10_hit15",
            }
            sel_obj = st.selectbox("최적화 목표", list(obj_options.keys()), key=f"obj_{market}")
            obj_key = obj_options[sel_obj]

            if st.button("📈 이 가중치로 성과 계산", key=f"calc_w_{market}"):
                with st.spinner("계산 중..."):
                    for top_k, label in [(10,"상위 10개"), (20,"상위 20개")]:
                        res = compute_custom_objective(df_rec, obj_key, manual_w, bench_col3, top_k)
                        if res:
                            avg_c = "#4ade80" if (res.get("avg",0) or 0) >= 0 else "#f87171"
                            # hit15 목표일 때 수치 의미 안내
                            if obj_key in ("hit15","top10_hit15"):
                                unit_desc = "20거래일 내 +15% 도달 비율(%)"
                                wr_desc   = f"날짜 승률 {res.get('win_rate',0):.0f}% (해당 날짜 도달 1개 이상)"
                            else:
                                unit_desc = "수익률(%)"
                                wr_desc   = f"승률 {res.get('win_rate',0):.0f}%"
                            st.markdown(
                                f'**{label}** ({unit_desc}): 평균 <span style="color:{avg_c};font-weight:900">{res.get("avg",0):+.2f}</span> | '
                                f'중앙 {res.get("median",0):+.2f} | {wr_desc} | '
                                f'최악 {res.get("worst",0):+.2f} | {res.get("n_dates",0)}개 날짜',
                                unsafe_allow_html=True,
                            )

        # ── Walk-forward 최적화 ───────────────────────────────────────
        with st.expander("🔄 Walk-forward 최적화 (과최적화 방지)", expanded=False):
            st.markdown(
                '<div style="font-size:12px;color:#475569;margin-bottom:10px">'
                '매 1개월마다 과거 6개월로 최적 가중치를 찾고, 다음 1개월에서 검증합니다. '
                'Test 성과가 꾸준히 플러스여야 진짜 예측력이 있습니다.</div>',
                unsafe_allow_html=True,
            )
            bench_col4 = "excess_market" if market == "KR" and "excess_market" in df_rec.columns else                          "excess_kospi" if market == "KR" and "excess_kospi" in df_rec.columns else "excess_spy"
            if st.button("▶ Walk-forward 최적화 실행", key=f"wf_run_{market}"):
                with st.spinner("Walk-forward 최적화 실행 중... (수 분 소요)"):
                    wf_results = walk_forward_optimization(
                        df_rec, train_months=6, test_months=1,
                        bench_col=bench_col4, objective=obj_key, top_k=10)
                    if wf_results:
                        wf_df = pd.DataFrame([{
                            "검증월":       r["test_period"],
                            "Train점수":    r["train_score"],
                            "Test점수":     r["test_score"],
                            "기본대비":     r["improvement"],
                            "Test승률%":    r["test_winrate"],
                            "Leader":       r["best_weights"]["leader"],
                            "Volume":       r["best_weights"]["volume"],
                            "Breakout":     r["best_weights"]["breakout"],
                            "Catalyst":     r["best_weights"]["catalyst"],
                            "Quality":      r["best_weights"]["quality"],
                            "Risk×":        r["best_weights"].get("risk_penalty", 0.08),
                        } for r in wf_results])
                        st.dataframe(wf_df, use_container_width=True, hide_index=True)

                        # 요약
                        tests = pd.Series([r["test_score"] for r in wf_results])
                        imps  = pd.Series([r["improvement"] for r in wf_results])
                        st.markdown(
                            f'**Walk-forward 요약** — 검증 {len(wf_results)}개월 | '
                            f'Test 평균: {tests.mean():+.3f} | '
                            f'기본대비 평균: {imps.mean():+.3f} | '
                            f'Test 승률: {(tests>0).mean()*100:.0f}%'
                        )

                        # 과최적화 경고
                        if imps.mean() < 0:
                            st.error("🔴 과최적화 의심 — Walk-forward Test 성과가 기본 가중치보다 낮습니다. 점수 로직 재검토 필요.")
                        elif imps.mean() < 0.01:
                            st.warning("⚠ Walk-forward 개선폭이 미미합니다.")
                        else:
                            st.success(f"✅ Walk-forward Test에서 기본 대비 평균 {imps.mean():+.3f} 개선")

                        # 안정적 평균 가중치 추천 (Risk 포함)
                        avg_w = {
                            "Leader":   round(wf_df["Leader"].mean(),   3),
                            "Volume":   round(wf_df["Volume"].mean(),   3),
                            "Breakout": round(wf_df["Breakout"].mean(), 3),
                            "Catalyst": round(wf_df["Catalyst"].mean(), 3),
                            "Quality":  round(wf_df["Quality"].mean(),  3),
                            "Risk×":    round(wf_df["Risk×"].mean(),    3),
                        }
                        st.markdown(f"**Walk-forward 평균 가중치 (추천):** `{avg_w}`")
                        wf_save_w = {
                            "leader": avg_w["Leader"], "volume": avg_w["Volume"],
                            "breakout": avg_w["Breakout"], "catalyst": avg_w["Catalyst"],
                            "quality": avg_w["Quality"], "risk_penalty": avg_w["Risk×"],
                            "improvement_test": round(float(imps.mean()), 4),
                            "objective": obj_key, "top_k": 10,
                        }
                        st.session_state[f"wf_avg_weights_{market}"] = wf_save_w
                    else:
                        st.warning("Walk-forward 실행 불가 — 데이터가 7개월 이상 필요합니다.")

            saved_w = st.session_state.get(f"wf_avg_weights_{market}")
            if saved_w:
                st.markdown(f'<div style="font-size:11px;color:#64748b">저장 대기 중인 Walk-forward 평균 가중치: {saved_w}</div>', unsafe_allow_html=True)
                if st.button(f"💾 이 Walk-forward 평균 가중치를 {market} 실전 가중치로 저장", key=f"wf_save_weights_{market}"):
                    saved = save_optimal_weights(saved_w, market=market, source="walk_forward_ui", objective=saved_w.get("objective", obj_key), top_k=int(saved_w.get("top_k", 10)))
                    st.success(f"✅ {market} 실전 가중치 저장 완료: {saved}")

    # ── 가중치 변화 이력 ─────────────────────────────────────────────
    history = load_weights_history()
    if history:
        st.markdown("---")
        st.markdown(
            '<div style="font-size:15px;font-weight:900;color:#f1f5f9;margin-bottom:8px">'
            '📈 가중치 최적화 이력</div>'
            '<div style="font-size:12px;color:#475569;margin-bottom:10px">'
            '매번 백테스트 후 최적화된 가중치 변화. 수렴하면 안정적인 모델.</div>',
            unsafe_allow_html=True,
        )

        # 미국/한국 분리
        for mkt, mkt_name in [("US","🇺🇸 미국"), ("KR","🇰🇷 한국")]:
            mkt_hist = [h for h in history if h.get("market","US")==mkt]
            if not mkt_hist: continue

            st.markdown(f"**{mkt_name} 가중치 이력 ({len(mkt_hist)}건)**")

            # 표 형태로 표시
            rows_html = (
                '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px">'
                '<tr style="color:#475569;border-bottom:1px solid rgba(255,255,255,0.06)">'
                '<th style="padding:5px 8px;text-align:left">날짜</th>'
                '<th style="padding:5px 8px;text-align:right">Leader</th>'
                '<th style="padding:5px 8px;text-align:right">Volume</th>'
                '<th style="padding:5px 8px;text-align:right">Breakout</th>'
                '<th style="padding:5px 8px;text-align:right">Catalyst</th>'
                '<th style="padding:5px 8px;text-align:right">Quality</th>'
                '<th style="padding:5px 8px;text-align:right">Risk×</th>'
                '<th style="padding:5px 8px;text-align:right">Train↑</th>'
                '<th style="padding:5px 8px;text-align:right">Test↑</th>'
                '<th style="padding:5px 8px;text-align:center">상태</th>'
                '</tr>'
            )
            for h in reversed(mkt_hist[-10:]):  # 최근 10건
                imp_tr = float(h.get("improvement_train",0) or 0)
                imp_te = float(h.get("improvement_test",0)  or 0)
                rejected = h.get("rejected", False)
                status = '<span style="color:#f87171">거부</span>' if rejected else '<span style="color:#4ade80">저장</span>'
                row_c  = "rgba(248,113,113,0.05)" if rejected else "rgba(0,0,0,0)"
                rows_html += (
                    f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:{row_c}">'
                    f'<td style="padding:5px 8px;color:#94a3b8">{h.get("date","")}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:#f59e0b">{h.get("leader",0):.2f}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:#60a5fa">{h.get("volume",0):.2f}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:#a78bfa">{h.get("breakout",0):.2f}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:#4ade80">{h.get("catalyst",0):.2f}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:#94a3b8">{h.get("quality",0):.2f}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:#f87171">{h.get("risk_penalty",0):.2f}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:{"#4ade80" if imp_tr>0 else "#f87171"}">{imp_tr:+.4f}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:{"#4ade80" if imp_te>0 else "#f87171"}">{imp_te:+.4f}</td>'
                    f'<td style="padding:5px 8px;text-align:center">{status}</td>'
                    f'</tr>'
                )
            rows_html += '</table></div>'
            st.markdown(rows_html, unsafe_allow_html=True)

            # plotly 라인 차트 (가중치 추이)
            try:
                import plotly.express as px
                import pandas as _pd
                df_hist = _pd.DataFrame(mkt_hist)
                df_hist = df_hist[~df_hist["rejected"]].tail(30)  # 거부된 것 제외, 최근 30건
                if len(df_hist) >= 2:
                    fig = px.line(
                        df_hist, x="date",
                        y=["leader","volume","breakout","catalyst","quality"],
                        labels={"value":"가중치","date":"날짜","variable":"요소"},
                        title=f"{mkt_name} 가중치 수렴 추이",
                        template="plotly_dark",
                        color_discrete_map={
                            "leader":"#f59e0b","volume":"#60a5fa",
                            "breakout":"#a78bfa","catalyst":"#4ade80","quality":"#94a3b8"
                        },
                        height=300,
                        markers=True,
                    )
                    fig.update_layout(
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font_color="#94a3b8", margin=dict(l=40,r=20,t=40,b=40),
                        legend=dict(orientation="h",y=-0.2),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass
    elif bt:
        st.markdown(
            '<div style="background:rgba(100,116,139,0.06);border-radius:7px;padding:10px 14px;'
            'font-size:12px;color:#475569;margin-top:8px">'
            '📈 가중치 이력은 최적화가 완료되면 여기에 누적됩니다.</div>',
            unsafe_allow_html=True,
        )

def _render_ai_insight(market: str):
    ai_key = "us_ai" if market=="US" else "kr_ai"
    leader = st.session_state.get("us_leader" if market=="US" else "kr_leader")
    if leader is None or isinstance(leader,tuple) or leader.empty:
        st.info("스크리너 실행 후 사용 가능합니다."); return

    prompts = generate_prompts(leader, market)
    for key,label in PROMPT_LABELS.items():
        with st.expander(label, expanded=False):
            st.text_area("복사 → GPT/Claude", prompts.get(key,""), height=250, key=f"ap_{market}_{key}")

    resp_type = st.selectbox("답변 유형", list(PROMPT_LABELS.keys()),
                              format_func=lambda x:PROMPT_LABELS[x], key=f"art_{market}")
    response  = st.text_area("AI 답변 붙여넣기", height=200, key=f"ar_{market}")
    c1,c2 = st.columns([3,1])
    with c1:
        if st.button("✅ 파싱 적용", use_container_width=True, type="primary", key=f"ap_btn_{market}"):
            if response.strip():
                parsed = parse_ai_response(response, leader, resp_type, market)
                existing = st.session_state.get(ai_key,{}); existing.update(parsed)
                st.session_state[ai_key]=existing
                st.success(f"{len(parsed)}개 파싱 완료") if parsed else st.warning("파싱 실패")
    with c2:
        if st.button("🗑 초기화", use_container_width=True, key=f"ac_btn_{market}"):
            st.session_state[ai_key]={}; st.rerun()


# ── 재무 ─────────────────────────────────────────────────────────

def _render_fundamental(top, fund, market):
    if top is None or isinstance(top,tuple) or top.empty:
        st.info("스크리너 실행 후 사용 가능합니다."); return
    if not fund:
        st.info("재무 데이터 없음"); return

    # 재무 데이터 수집 상태 확인
    n_available = sum(1 for v in fund.values() if v.get("data_available",False))
    if n_available == 0:
        st.warning(
            "⚠️ 재무 데이터 없음 — Yahoo Finance 수집 실패.\n\n"
            "로컬 PC의 일반 인터넷 환경에서 실행 시 정상 수집됩니다.\n"
            "(회사 내부 네트워크, VPN, 방화벽 환경에서 Yahoo Finance 차단될 수 있음)"
        )
    elif n_available < len(fund):
        st.info(f"일부 재무 데이터만 수집됨: {n_available}/{len(fund)}개")

    sort_by = st.selectbox(
        "정렬",
        ["재무점수","라이브 보정점수(final_score)","매출성장률","영업이익성장률"],
        key=f"fs_{market}",
        help="final_score는 factor 계산 뒤 재무/테마/컨텍스트/회피 보정이 반영된 라이브 조정 점수입니다.",
    )
    sort_map = {"재무점수":"fund_score","라이브 보정점수(final_score)":"final_score","매출성장률":"rev_growth","영업이익성장률":"op_growth"}

    rows=[]
    for tk in top.index:
        f=fund.get(tk,{}); r=top.loc[tk]
        rows.append({
            "ticker":tk,"name":r.get("name",r.get("company",tk)),
            "final_score":float(r.get("final_score",0)),
            "fund_score":float(f.get("fund_score") or 0) if f.get("data_available") else None,
            "fund_grade":f.get("fund_grade","—") if f.get("data_available") and f.get("fund_grade") else "N/A",
            "rev_growth":f.get("rev_growth",float("nan")),
            "op_growth":f.get("op_growth",float("nan")),
            "eps_growth":f.get("eps_growth",float("nan")),
            "roe":f.get("roe",float("nan")),
            "op_margin":f.get("op_margin",float("nan")),
            "forward_pe":f.get("forward_pe",float("nan")),
            "rev_accel":f.get("rev_accel",False),
        })
    sk=sort_map[sort_by]
    def _safe_sort(x):
        v = x.get(sk)
        if v is None: return -999
        try: return float(v) if math.isfinite(float(v)) else -999
        except: return -999
    rows.sort(key=_safe_sort, reverse=True)

    for i,r in enumerate(rows,1):
        fs_raw = r.get("fund_score")
        fs = float(fs_raw) if fs_raw is not None else None
        has_fund = fs is not None
        c = ("#4ade80" if fs>=70 else "#818cf8" if fs>=55 else "#fbbf24" if fs>=40 else "#f87171") if has_fund else "#334155"
        accel='<span style="color:#4ade80;font-size:11px;font-weight:700">🚀가속</span>' if r["rev_accel"] else ""
        name=f'{r["name"]}({r["ticker"]})' if market=="KR" else r["ticker"]

        def fv(v,d=1,s="%"):
            try: v=float(v)
            except: return "—"
            return "—" if not math.isfinite(v) else f"{'+'if v>=0 else ''}{v:.{d}f}{s}"

        st.markdown(
            f'<div style="background:#0f172a;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:10px 14px;margin-bottom:7px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            f'<div style="display:flex;gap:8px;align-items:center"><span style="font-size:12px;color:#475569">#{i}</span>'
            f'<span style="font-size:15px;font-weight:900;color:#f1f5f9">{name}</span>{accel}</div>'
            f'<div style="display:flex;gap:10px;align-items:center">'
            f'<span style="background:{c}22;color:{c};border:1px solid {c}44;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:900">{r["fund_grade"] or "N/A"}</span>'
            f'<span title="final_score: 재무/테마/컨텍스트/회피 보정 후 라이브 조정 점수입니다." style="font-size:12px;color:#475569">라이브보정 {r["final_score"]:.0f}</span></div></div>'
            f'<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:5px;font-size:11px;text-align:center">'
            + "".join(
                f'<div style="background:rgba(0,0,0,0.2);border-radius:5px;padding:5px">'
                f'<div style="color:#334155;margin-bottom:2px">{lb}</div>'
                f'<div style="font-weight:700;color:{"#4ade80" if ip and (lambda v: v>0 if math.isfinite(float(v or 0)) else False)(r.get(k)) else "#94a3b8"}">{fv(r.get(k),d,sf)}</div></div>'
                for lb,k,ip,d,sf in [
                    ("매출YoY","rev_growth",True,1,"%"),("영업YoY","op_growth",True,1,"%"),
                    ("EPS YoY","eps_growth",True,1,"%"),("ROE","roe",True,1,"%"),
                    ("영업률","op_margin",True,1,"%"),("FwdPE","forward_pe",False,1,"x"),
                    ("재무점수","fund_score",False,0,""),
                ]
            )
            + '</div></div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════
# 스크리너 실행 함수들
# ════════════════════════════════════════════════════════════════════

def run_us_screener(top_n=30, save_charts=True, force_refresh=False):
    if force_refresh:
        cleared=clear_all_cache(); st.info(f"캐시 {cleared}개 삭제")

    st.markdown("#### 🇺🇸 미국 수집 중...")
    us_tickers=get_us_tickers()
    st.info(f"미국 {len(us_tickers)}개 티커")
    pb=st.progress(0); tx=st.empty()
    def up(c,t,tk): pb.progress(int(c/t*100)); tx.caption(f"US ({c}/{t}): {tk}")
    us_data,us_fail=fetch_us_universe(us_tickers, progress_cb=up)
    pb.empty(); tx.empty()

    us_tt=st.session_state.get("us_theme_tickers",[])
    if us_tt:
        missing=[t for t in us_tt if t not in us_data]
        if missing:
            with st.spinner(f"테마 종목 {len(missing)}개 추가 수집..."):
                extra,_=fetch_us_universe(missing); us_data.update(extra)

    us_leader=us_buyable=us_breakout=us_turnaround=us_full_df=None
    if len(us_data)>=20:
        with st.spinner("미국 팩터 계산 (4트랙 + 재무)..."):
            us_leader,us_buyable,us_breakout,us_turnaround,us_excl,us_full_df=run_us_pipeline(
                us_data,top_n=top_n,
                theme_tickers=us_tt,
                saved_themes=st.session_state.get("us_themes",[]),
                integrate_fundamentals=True,
                context_factors=st.session_state.get("us_context_factors",{}),
                avoid_tickers=st.session_state.get("us_avoid_tickers",[]),
            )
        st.success(f"✅ 미국 — Leader {_n(us_leader)} / Buyable {_n(us_buyable)} / Breakout {_n(us_breakout)} / Turnaround {_n(us_turnaround)}")
        # 버려진 종목을 '수집 실패'와 '필터 탈락'으로 구분 표시
        _collect_fail = list(dict.fromkeys(us_fail))        # yfinance 수집 실패 (상폐/먹통)
        _filter_drop  = list(dict.fromkeys(us_excl))        # 필터/팩터 탈락
        with st.expander(f"📋 분석 요약 — 유니버스 {len(us_tickers)}개 중 수집 {len(us_data)}개 · 제외 {len(_collect_fail)+len(_filter_drop)}개", expanded=False):
            if _collect_fail:
                st.markdown(f"**🔌 수집 실패 {len(_collect_fail)}개** (상폐/합병/먹통 티커 — 무시해도 됨)")
                st.caption(", ".join(_collect_fail[:50]) + (" …" if len(_collect_fail) > 50 else ""))
            if _filter_drop:
                st.markdown(f"**🚫 필터/팩터 탈락 {len(_filter_drop)}개** (가격<$5·시총<$2B·거래대금<$50M 등)")
                st.caption(", ".join(_filter_drop[:50]) + (" …" if len(_filter_drop) > 50 else ""))
            if us_full_df is not None and not isinstance(us_full_df, tuple):
                st.markdown(f"**✅ 최종 분석(팩터 계산) {len(us_full_df)}개** → 트랙 분류 후 화면 표시")
        us_fail += us_excl   # 세션 저장용으로 합침 (표시 이후)
        # 분석 종목 수 세션 저장 (사이드바에 항상 표시용)
        if us_full_df is not None and not isinstance(us_full_df, tuple):
            st.session_state["us_analyzed_count"] = len(us_full_df)
            st.session_state["us_universe_count"] = len(us_tickers)

        if us_full_df is not None and not isinstance(us_full_df,tuple):
            us_themes=st.session_state.get("us_themes",[])
            if us_themes:
                ts_list=[]
                for th in us_themes:
                    ts=calc_theme_score(th,us_full_df)
                    ts.update({k:th.get(k,"") for k in ["name","stage","conviction","reason","tickers"]})
                    ts_list.append(ts)
                ts_list.sort(key=lambda x:x["theme_score"],reverse=True)
                st.session_state.us_theme_scores=ts_list

            from screener.engine import analyze_sector_etfs as _ase
            bench_close={k:v["ohlcv"]["Close"] for k,v in us_data.items() if is_us_live_etf_ticker(k) and "ohlcv" in v}
            st.session_state.sector_analysis=_ase(bench_close)
            st.session_state.universe_themes_us=calc_universe_theme_strength(us_full_df,"US")

        if save_charts:
            with st.spinner("차트 저장..."):
                for tk in (us_leader.index[:15] if _n(us_leader) else []):
                    if tk in us_data: save_chart(us_data[tk]["ohlcv"],tk,us_data[tk].get("company",""))

    us_fund=_extract_fund_from_df(us_full_df)

    # 재무 수집 결과 알림
    us_fund_ok = sum(1 for v in us_fund.values() if v.get("data_available",False))
    if us_fund_ok > 0:
        st.success(f"✅ 미국 재무 수집: {us_fund_ok}/{len(us_fund)}개")
    elif us_full_df is not None and not isinstance(us_full_df,tuple) and len(us_fund)>0:
        st.warning(
            "⚠️ 미국 재무 데이터 수집 실패 — Yahoo Finance Rate Limit\n\n"
            "집 인터넷에서도 단시간 다량 요청 시 일시적으로 차단됩니다.\n\n"
            "**해결:**\n"
            "1. 5~10분 후 **↺ 일반 초기화** 후 재실행 (가장 흔한 해결)\n"
            "2. 그래도 안 되면 **🔥 완전 초기화** 후 재실행\n\n"
            "재무 없어도 가격/거래량/RS 기반 점수는 정상 동작합니다."
        )

    st.session_state.update({
        "us_leader":us_leader,"us_buyable":us_buyable,
        "us_breakout":us_breakout,"us_turnaround":us_turnaround,
        "us_full_df":us_full_df,"us_fund":us_fund,
        "failed_us":us_fail,"scr_date":TODAY,
        "regime_us":(us_full_df.attrs.get("regime") if (us_full_df is not None and not isinstance(us_full_df,tuple)) else None),
    })

    # 애널리스트 — Leader+Breakout+Turnaround+Buyable+테마 합산 후 중복제거 최대 40개
    us_analyst={}
    _collect_targets=set()
    if _n(us_leader):    _collect_targets.update(us_leader.index.tolist()[:15])
    if _n(us_breakout):  _collect_targets.update(us_breakout.index.tolist()[:10])
    if _n(us_turnaround):_collect_targets.update(us_turnaround.index.tolist()[:10])
    if _n(us_buyable):   _collect_targets.update(us_buyable.index.tolist()[:10])
    _collect_targets.update(st.session_state.get("us_theme_tickers",[]))
    _collect_targets = [t for t in _collect_targets if not is_us_backtest_excluded_ticker(t)]
    _collect_targets = list(_collect_targets)[:40]
    if _collect_targets:
        st.markdown(f"#### 🎯 미국 애널리스트 수집 ({len(_collect_targets)}개: Leader+Breakout+Turnaround+테마)...")
        pb2=st.progress(0); tx2=st.empty()
        def apcb(c,t,tk): pb2.progress(int(c/t*100)); tx2.caption(f"애널리스트 {tk}")
        us_analyst=fetch_analyst_batch(_collect_targets, apcb)
        pb2.empty(); tx2.empty()
        st.success(f"✅ 애널리스트 {len(us_analyst)}개")

    # 백테스트
    backtest_result=None
    if us_data and len(us_data)>=20:
        with st.spinner("📊 백테스트... (캐시 있으면 즉시)"):
            cached_bt=load_backtest()
            if cached_bt:
                backtest_result=cached_bt; st.success("✅ 백테스트 캐시 로드")
            else:
                pb3=st.progress(0); tx3=st.empty()
                def btcb(c,t,tk): pb3.progress(int(c/t*100)); tx3.caption(f"BT {tk}")
                backtest_result=run_walkforward_backtest(us_data,lookahead=20,step=5,progress_cb=btcb)
                pb3.empty(); tx3.empty()
                if backtest_result:
                    ov=backtest_result.get("overall",{})
                    st.success(f"✅ BT — 평균 {ov.get('avg_ret',0):+.1f}% | 승률 {ov.get('win_rate',0):.0f}% | {backtest_result.get('n_records',0)}건")
                    if st.session_state.get("sidebar_auto_save_weights", False):
                        with st.spinner("⚙️ 가중치 최적화..."):
                            opt_w=optimize_weights(backtest_result)
                            imp_train = float(opt_w.get("improvement_train", opt_w.get("improvement", 0)) or 0)
                            imp_test  = float(opt_w.get("improvement_test",  0) or 0)
                            rejected  = opt_w.get("rejected_reason","")
                            if rejected:
                                st.warning(f"⚠ {rejected} — 기본 가중치 유지")
                            elif imp_train > 0.01 or imp_test > 0.01:
                                st.info(
                                    f"🎯 최적 가중치 저장 — "
                                    f"Train +{imp_train:.4f} / Test +{imp_test:.4f} Sharpe"
                                )
                            else:
                                st.info(f"기본 가중치가 최적에 가깝습니다 (Train {imp_train:+.4f} / Test {imp_test:+.4f})")
                        st.markdown('<div style="font-size:11px;color:#64748b;padding:6px 10px;background:rgba(0,0,0,0.2);border-radius:5px">💡 최적 가중치는 다음 실행부터 반영됩니다.</div>', unsafe_allow_html=True)
                    else:
                        st.info("자동 가중치 최적화/저장은 꺼져 있습니다. 고급/진단 탭의 Walk-forward 결과에서 직접 저장하세요.")

    # 스냅샷 백테스트 (워크포워드 완료 후 추가 실행)
    snapshot_results = []
    if backtest_result and us_data and len(us_data) >= 20:
        with st.spinner("📸 스냅샷 백테스트 (과거 날짜 복원)..."):
            from screener.backtest import run_snapshot_backtest
            snapshot_results = run_snapshot_backtest(us_data, top_k=20, lookahead=20)
            if snapshot_results:
                best = max(snapshot_results, key=lambda x: x.get("avg_ret",0))
                st.success(f"✅ 스냅샷 {len(snapshot_results)}개 날짜 · 최고: {best['snapshot_date']} avg {best['avg_ret']:+.1f}%")

    st.session_state.update({
        "us_analyst":us_analyst,
        "backtest_result":backtest_result,
        "snapshot_results":snapshot_results,
    })
    try:
        if _n(us_leader): save_csv(us_leader,"us_leader")
    except Exception: pass


def run_kr_screener(top_n=30, save_charts=True, force_refresh=False):
    if force_refresh:
        from screener.engine import clear_all_cache as cac
        cleared=cac(); st.info(f"캐시 {cleared}개 삭제")

    st.markdown("#### 🇰🇷 한국 수집 중...")
    kr_tickers=get_kr_tickers()
    st.info(f"한국 {len(kr_tickers)}개 티커")
    pb=st.progress(0); tx=st.empty()
    def up(c,t,tk): pb.progress(int(c/t*100)); tx.caption(f"KR ({c}/{t}): {tk}")
    kr_data,kr_fail=fetch_kr_universe(kr_tickers,progress_cb=up)
    pb.empty(); tx.empty()

    kr_tt=st.session_state.get("kr_theme_tickers",[])
    if kr_tt:
        missing=[t for t in kr_tt if t not in kr_data]
        if missing:
            with st.spinner(f"테마 {len(missing)}개 추가..."):
                extra,_=fetch_kr_universe(missing); kr_data.update(extra)

    kr_leader=kr_buyable=kr_breakout=kr_turnaround=kr_full_df=None
    if len(kr_data)>=10:
        kr_bench=fetch_kr_benchmarks()
        with st.spinner("한국 팩터 계산..."):
            kr_leader,kr_buyable,kr_breakout,kr_turnaround,kr_excl,kr_full_df=run_kr_pipeline(
                kr_data,kr_bench,top_n=top_n,
                theme_tickers=kr_tt,
                saved_themes=st.session_state.get("kr_themes",[]),
                integrate_fundamentals=True,
                context_factors=st.session_state.get("kr_context_factors",{}),
                avoid_tickers=st.session_state.get("kr_avoid_tickers",[]),
            )
        kr_fail+=kr_excl
        st.success(f"✅ 한국 — Leader {_n(kr_leader)} / Buyable {_n(kr_buyable)} / Breakout {_n(kr_breakout)} / Turnaround {_n(kr_turnaround)}")
        # 분석 요약 (미국과 통일)
        _kr_collect_fail = list(dict.fromkeys(kr_fail))
        with st.expander(f"📋 분석 요약 — 수집 {len(kr_data)}개 · 제외 {len(_kr_collect_fail)}개", expanded=False):
            if _kr_collect_fail:
                st.markdown(f"**🚫 수집/필터 제외 {len(_kr_collect_fail)}개**")
                st.caption(", ".join(_kr_collect_fail[:50]) + (" …" if len(_kr_collect_fail) > 50 else ""))
            if kr_full_df is not None and not isinstance(kr_full_df, tuple):
                st.markdown(f"**✅ 최종 분석(팩터 계산) {len(kr_full_df)}개** → 트랙 분류 후 화면 표시")
        # 분석 종목 수 세션 저장 (사이드바에 항상 표시용)
        if kr_full_df is not None and not isinstance(kr_full_df, tuple):
            st.session_state["kr_analyzed_count"] = len(kr_full_df)
            st.session_state["kr_universe_count"] = len(kr_tickers)
        if kr_full_df is not None and not isinstance(kr_full_df,tuple):
            st.session_state.universe_themes_kr=calc_universe_theme_strength(kr_full_df,"KR")
        if save_charts:
            with st.spinner("차트 저장..."):
                for tk in (kr_leader.index[:15] if _n(kr_leader) else []):
                    if tk in kr_data: save_chart(kr_data[tk]["ohlcv"],tk,kr_data[tk].get("name",""))

        # 벤치마크 데이터를 kr_data에 합쳐서 백테스트에 KOSPI/KOSDAQ 지수 전달
        # fetch_kr_benchmarks()는 Series를 반환하므로 백테스트 입력 규격(DataFrame Close 컬럼)으로 변환합니다.
        kr_bt_data = {**kr_data}
        for code, close_s in kr_bench.items():
            if code not in kr_bt_data:
                if isinstance(close_s, pd.Series):
                    bench_ohlcv = close_s.to_frame("Close")
                else:
                    bench_ohlcv = close_s
                kr_bt_data[code] = {"ohlcv": bench_ohlcv, "name": f"벤치마크_{code}", "market": "INDEX"}
        st.session_state["_kr_bt_data"] = kr_bt_data  # 백테스트용 저장

    kr_fund=_extract_fund_from_df(kr_full_df)
    st.session_state.update({
        "kr_leader":kr_leader,"kr_buyable":kr_buyable,
        "kr_breakout":kr_breakout,"kr_turnaround":kr_turnaround,
        "kr_full_df":kr_full_df,"kr_fund":kr_fund,
        "failed_kr":kr_fail,"scr_date":TODAY,
        "regime_kr":(kr_full_df.attrs.get("regime") if (kr_full_df is not None and not isinstance(kr_full_df,tuple)) else None),
    })
    # 한국 워크포워드 백테스트 + 가중치 최적화
    kr_backtest_result = None
    if kr_data and len(kr_data) >= 10:
        # 백테스트 가능 종목 수 미리 확인
        from screener.backtest import KR_MIN_BT_PRICE, KR_MIN_BT_AVG_AMOUNT
        def _kr_bt_precheck(v, min_hist=120):
            try:
                df = v.get("ohlcv")
                name = str(v.get("name", ""))
                if df is None or len(df) < min_hist + 20 + 5:
                    return False
                if is_kr_precheck_excluded_security_name(name):
                    return False
                if float(df["Close"].iloc[-1]) < KR_MIN_BT_PRICE:
                    return False
                amt = float(df["Amount"].tail(20).mean()) if "Amount" in df.columns else float("nan")
                if math.isfinite(amt) and amt < KR_MIN_BT_AVG_AMOUNT:
                    return False
                return True
            except Exception:
                return False
        bt_eligible = [
            t for t,v in kr_data.items()
            if not is_kr_backtest_excluded_ticker(t)
            and not is_kr_benchmark_code(t)
            and "ohlcv" in v
            and _kr_bt_precheck(v, 120)
        ]
        st.info(
            f"📊 KR 백테스트 준비: 전체 {len(kr_data)}개 → "
            f"ETF·저가주·거래대금부족·이력부족 제외 후 **{len(bt_eligible)}개** 대상"
        )
        if len(bt_eligible) < 5:
            st.warning(
                f"⚠ 백테스트 가능 종목이 {len(bt_eligible)}개로 너무 적습니다.\n\n"
                f"필터: 120거래일 이상, 1,000원 이상, 최근 20일 평균 거래대금 10억 원 이상, ETF/ETN/SPAC/우선주 제외."
            )
        with st.spinner("📊 한국 백테스트... (캐시 있으면 즉시)"):
            cached_kr_bt = load_kr_backtest()
            if cached_kr_bt:
                kr_backtest_result = cached_kr_bt
                st.success("✅ 한국 백테스트 캐시 로드")
            else:
                pb_kr=st.progress(0); tx_kr=st.empty()
                def kr_btcb(c,t,tk): pb_kr.progress(int(c/t*100)); tx_kr.caption(f"KR BT {tk}")
                # kr_bt_data = kr_data + KOSPI200 벤치마크
                kr_bt_data = st.session_state.get("_kr_bt_data", kr_data)
                # 이력 길이에 따라 min_history 자동 조정
                max_hist = max(
                    (len(v["ohlcv"]) for v in kr_bt_data.values() if "ohlcv" in v),
                    default=0
                )
                if max_hist >= 300:   kr_min_hist = 120
                elif max_hist >= 160: kr_min_hist = 80
                else:                 kr_min_hist = 60
                st.caption(f"KR 백테스트 min_history={kr_min_hist}, step=5 (최대 이력 {max_hist}거래일)")
                kr_backtest_result = run_kr_walkforward_backtest(
                    kr_bt_data, lookahead=20, step=5,
                    min_history=kr_min_hist, bench_code="kospi", progress_cb=kr_btcb)
                pb_kr.empty(); tx_kr.empty()
                # 한국 결과에 market 태그 명시 (미국 결과와 구분)
                if kr_backtest_result:
                    kr_backtest_result["market"] = "KR"
                    ov_kr = kr_backtest_result.get("overall",{})
                    st.success(
                        f"✅ KR BT — 평균 {ov_kr.get('avg_ret',0):+.1f}% | "
                        f"승률 {ov_kr.get('win_rate',0):.0f}% | "
                        f"{kr_backtest_result.get('n_records',0)}건"
                    )
                    if st.session_state.get("sidebar_auto_save_weights", False):
                        with st.spinner("⚙️ 한국 가중치 최적화..."):
                            opt_kr = optimize_kr_weights(kr_backtest_result)
                            imp_tr = float(opt_kr.get("improvement_train",0) or 0)
                            imp_te = float(opt_kr.get("improvement_test",0)  or 0)
                            if opt_kr.get("rejected_reason"):
                                st.warning(f"⚠ {opt_kr['rejected_reason']} — 기본 가중치 유지")
                            else:
                                st.info(f"🎯 KR 가중치 저장 — Train +{imp_tr:.4f} / Test +{imp_te:.4f}")
                        st.markdown('<div style="font-size:11px;color:#64748b;padding:6px 10px;background:rgba(0,0,0,0.2);border-radius:5px">💡 KR 최적 가중치는 다음 실행부터 반영됩니다.</div>', unsafe_allow_html=True)
                    else:
                        st.info("KR 자동 가중치 최적화/저장은 꺼져 있습니다. 고급/진단 탭의 Walk-forward 결과에서 직접 저장하세요.")
                else:
                    st.warning(
                        "⚠ KR 백테스트 결과가 비었습니다. 종목별 OHLCV가 최소 "
                        f"{kr_min_hist + 20 + 5}거래일 이상이고, 저가주/거래대금/이상값 필터를 통과하는지 확인하세요. "
                        "캐시가 오래된 데이터라면 '전체 캐시 삭제' 또는 '↺ KR' 후 다시 실행하세요."
                    )

    st.session_state["backtest_kr_result"] = kr_backtest_result

    try:
        if _n(kr_leader): save_csv(kr_leader,"kr_leader")
    except Exception: pass


def run_screener(top_n=30, save_charts=True, force_refresh=False):
    run_us_screener(top_n=top_n, save_charts=save_charts, force_refresh=force_refresh)
    run_kr_screener(top_n=top_n, save_charts=save_charts, force_refresh=False)


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="MU식 리레이팅 스크리너",
        page_icon="🎯", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&display=swap');
    html,body,[class*="css"]{font-family:'Noto Sans KR',sans-serif;}
    .stApp{background:#020817;color:#e2e8f0;}
    .block-container{padding-top:0.8rem;max-width:1380px;}
    h1,h2,h3,h4{color:#f1f5f9!important;}
    p,li,span,div{font-size:14px!important;}
    .stButton>button{border-radius:8px!important;font-weight:800!important;}
    .stTabs [data-baseweb="tab"]{color:#64748b!important;font-weight:700;}
    .stTabs [aria-selected="true"]{color:#f1f5f9!important;}
    .stRadio label{font-size:13px!important;font-weight:700!important;}
    hr{border-color:rgba(255,255,255,0.06)!important;}
    </style>
    """, unsafe_allow_html=True)

    init_session()

    # ── 헤더 ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.06)">'
        '<div style="width:42px;height:42px;background:linear-gradient(135deg,#f59e0b,#ef4444);border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:22px">🎯</div>'
        '<div><div style="font-size:20px;font-weight:900;color:#f1f5f9">MU식 리레이팅 스크리너</div>'
        '<div style="font-size:12px;color:#475569">오늘 비싸보여도 계속 리레이팅될 종목은 무엇인가?</div></div></div>',
        unsafe_allow_html=True,
    )

    # ── 사이드바 ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ 설정")

        # 데이터 기준일 배지
        _dd = get_data_date()
        if _dd:
            _age = data_age_days()
            if _age == 0:
                st.success(f"📅 데이터 기준일: {_dd} (오늘)")
            else:
                st.warning(f"📅 데이터 기준일: {_dd} ({_age}일 전)\n\n최신화: ⚙️고급/진단 탭 → 🔥 완전 초기화 후 재스캔")
        else:
            st.info("📅 데이터 없음 — 스캔하면 오늘 데이터로 수집됩니다")

        # 시장 선택
        _saved_mkt = st.session_state.get("market", "US")
        if _saved_mkt not in ("US", "KR"):
            _saved_mkt = "US"   # 과거 US+KR 저장값 방어
        market = st.radio("🌐 시장", ["US", "KR"],
                          horizontal=True, key="market_selector",
                          index=["US", "KR"].index(_saved_mkt))
        st.session_state["market"] = market
        mkt = market

        # 분석 종목 수 항상 표시 (마지막 스캔 기준)
        _ac = st.session_state.get(f"{market.lower()}_analyzed_count")
        _uc = st.session_state.get(f"{market.lower()}_universe_count")
        if _ac:
            if _uc:
                st.caption(f"📊 {market} 분석 종목: **{_ac}개** (유니버스 {_uc}개 중)")
            else:
                st.caption(f"📊 {market} 분석 종목: **{_ac}개**")
        else:
            st.caption(f"📊 {market} 아직 스캔 안 됨")

        top_n       = st.slider("Top N", 10, 50, 25, 5, key="sidebar_top_n")
        save_charts = st.checkbox("차트 저장", value=True, key="sidebar_save_charts")
        st.checkbox("백테스트 후 자동 가중치 최적화/저장", value=False, key="sidebar_auto_save_weights",
                    help="꺼두면 기존 실전 가중치를 덮어쓰지 않습니다. Walk-forward 결과에서 수동 저장하는 방식을 권장합니다.")
        st.markdown("---")

        # 상태
        us_ok = _n(st.session_state.get("us_leader")) > 0
        kr_ok = _n(st.session_state.get("kr_leader")) > 0
        st.markdown(
            f'<div style="font-size:12px;line-height:2">'
            f'{"✅" if us_ok else "⬜"} 미국 {f"({_n(st.session_state.get(chr(117)+chr(115)+chr(95)+chr(108)+chr(101)+chr(97)+chr(100)+chr(101)+chr(114)))}개)" if us_ok else ""}<br>'
            f'{"✅" if kr_ok else "⬜"} 한국 {f"({_n(st.session_state.get(chr(107)+chr(114)+chr(95)+chr(108)+chr(101)+chr(97)+chr(100)+chr(101)+chr(114)))}개)" if kr_ok else ""}'
            + (f'<br><span style="color:#475569;font-size:11px">{st.session_state.scr_date}</span>' if st.session_state.scr_date else "")
            + '</div>', unsafe_allow_html=True,
        )
        theme_us = len(st.session_state.get("us_theme_tickers",[]))
        theme_kr = len(st.session_state.get("kr_theme_tickers",[]))
        if theme_us or theme_kr:
            st.markdown(f'<div style="font-size:11px;color:#f59e0b">🎯 테마: US {theme_us} / KR {theme_kr}개</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Load latest saved results if you do not want to rerun the screener.")
        latest_file = _latest_saved_result_file(market)
        if latest_file:
            st.caption(f"최근 저장 결과: `{latest_file.name}`")
        else:
            st.caption(f"{market} 저장 결과 파일 없음")
        if st.button("📂 Load latest saved results", use_container_width=True, key=f"btn_load_latest_{market}"):
            ok, msg = _load_saved_results(market)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.info(msg)

        st.markdown("---")

        # 실행 버튼
        c1,c2 = st.columns(2)
        with c1:
            if st.button("🇺🇸 미국", use_container_width=True, type="primary", key="btn_us"):
                run_us_screener(top_n=top_n, save_charts=save_charts); st.rerun()
        with c2:
            if st.button("↺ US", use_container_width=True, key="btn_us_rf"):
                run_us_screener(top_n=top_n, save_charts=save_charts, force_refresh=True); st.rerun()
        c3,c4 = st.columns(2)
        with c3:
            if st.button("🇰🇷 한국", use_container_width=True, type="primary", key="btn_kr"):
                run_kr_screener(top_n=top_n, save_charts=save_charts); st.rerun()
        with c4:
            if st.button("↺ KR", use_container_width=True, key="btn_kr_rf"):
                run_kr_screener(top_n=top_n, save_charts=save_charts, force_refresh=True); st.rerun()
        st.markdown('<div style="font-size:11px;color:#334155;margin:3px 0">— 또는 —</div>', unsafe_allow_html=True)
        if st.button("🌐 미국+한국 동시", use_container_width=True, key="btn_both"):
            run_screener(top_n=top_n, save_charts=save_charts); st.rerun()

        # 캐시 2단계 분리
        st.markdown('<div style="font-size:11px;color:#475569;margin:8px 0 4px">🗑 캐시 관리</div>', unsafe_allow_html=True)
        _dd2 = get_data_date()
        _age2 = data_age_days()
        _datetxt = (f"{_dd2} ({'오늘' if _age2==0 else f'{_age2}일 전'})" if _dd2 else "없음")
        st.caption(f"📅 데이터 기준일: {_datetxt}  ·  저장 위치: `{DATA_DIR}`")
        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            if st.button("⚡ 점수만 재계산", use_container_width=True, key="btn_cache_scores",
                         help="가격/수집 데이터는 그대로 두고 점수·팩터·백테스트만 다시 계산.\n가중치·팩터 로직을 바꿨을 때 사용 — 한국 재수집(수십 분)을 건너뛴다."):
                from screener.backtest import clear_all_backtest_cache
                n1=clear_scores_cache(); n2=clear_all_backtest_cache()
                st.success(f"점수/백테스트 {n1+n2}개 삭제 (수집 데이터 유지)"); st.rerun()
        with cc2:
            if st.button("↺ 일반 초기화", use_container_width=True, key="btn_cache_normal",
                         help="오늘 가격/팩터 캐시 삭제 (재수집 발생)"):
                n=clear_all_cache(); st.success(f"{n}개 삭제"); st.rerun()
        with cc3:
            if st.button("🔥 완전 초기화", use_container_width=True, key="btn_cache_full",
                         help="백테스트·가중치·스냅샷까지 전부 삭제 (재수집 발생)"):
                from screener.backtest import clear_all_backtest_cache
                n1=clear_all_cache(); n2=clear_all_backtest_cache()
                st.success(f"전체 {n1+n2}개 삭제"); st.rerun()

        st.markdown("---")
        st.markdown(
            '<div style="font-size:11px;color:#334155;line-height:1.9">'
            '🎯 <b>리레이팅 후보</b>: 비싸도 추정치·테마로 더 갈 종목<br>'
            '🔥 Leader = 이미 강한 주도주<br>'
            '⚡ Breakout = 막 돌파하는 종목<br>'
            '🔄 Turnaround = 저점 반등 초기<br>'
            '⚠ <b>Buyable ≠ 리레이팅</b>: 싼 자리와 MU식은 다름</div>',
            unsafe_allow_html=True,
        )

    # ── 메인 탭 ───────────────────────────────────────────────────
    tabs = st.tabs(["🎯 리레이팅 후보", "🤖 GPT 리레이팅 분석", "💼 내 포트폴리오", "⚙️ 고급/진단"])

    with tabs[0]:
        render_rerating_tab(market)

    with tabs[1]:
        render_gpt_analysis_tab(market)

    with tabs[2]:
        render_portfolio_tab()

    with tabs[3]:
        # 사이드바에서 이미 시장 선택 → 고급 탭은 그대로 사용
        adv_mkt = "KR" if st.session_state.get("market","US") == "KR" else "US"
        render_advanced_tab(adv_mkt)

    st.markdown(
        '<div style="font-size:11px;color:#1e293b;margin-top:14px">'
        '⚠️ 본 스크리너는 정량 후보 압축 도구이며 투자 추천이 아닙니다. '
        '최종 매수 판단은 직접 확인 후 본인 책임으로 이루어져야 합니다.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
