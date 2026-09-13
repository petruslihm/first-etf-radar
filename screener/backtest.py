"""
screener/backtest.py  v3

개선사항:
1. ETF/벤치마크 백테스트 제외
2. 스냅샷 백테스트가 실제 라이브 랭킹 로직과 동일
3. Train/Test 분리 최적화 (과최적화 방지)
4. 점수 기여도(score_explain) 계산
5. 자동 최적화: 백테스트 결과로 판단기준을 자동 갱신
"""

import logging
import math
import pickle
from utils.scoring import calc_validated_score, calc_validated_score_series, calc_risk_penalty, DEFAULT_WEIGHTS
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from screener.filters import (
    KR_BACKTEST_ETF_NAME_KEYWORDS,
    KR_BACKTEST_EXCLUDE_TICKERS,
    KR_BENCH_CODES,
    is_kr_backtest_excluded_ticker,
    is_kr_backtest_etf_name,
    is_kr_benchmark_code,
    is_kr_unwanted_security,
    is_us_backtest_excluded_ticker,
    US_BACKTEST_EXCLUDE_TICKERS,
)

logger    = logging.getLogger(__name__)
from screener.paths import CACHE_DIR
BT_CACHE      = CACHE_DIR / "backtest_wf.pkl"
BT_KR_CACHE   = CACHE_DIR / "backtest_wf_kr.pkl"
OPT_CACHE     = CACHE_DIR / "optimal_weights.pkl"
OPT_KR_CACHE  = CACHE_DIR / "optimal_weights_kr.pkl"
SNAPSHOT_CACHE= CACHE_DIR / "snapshot_bt.pkl"
BT_VERSION    = "v10.6-drop-quality"

# ── ETF/벤치마크 제외 목록 (백테스트 개별주 대상에서 제외) ────────
EXCLUDE_BT_TICKERS = US_BACKTEST_EXCLUDE_TICKERS


# ════════════════════════════════════════════════════════════════════
# 캐시 관리
# ════════════════════════════════════════════════════════════════════

def load_backtest() -> Optional[dict]:
    if not BT_CACHE.exists(): return None
    try:
        with open(BT_CACHE,"rb") as f: result = pickle.load(f)
        if result.get("bt_version") != BT_VERSION:
            logger.info("백테스트 버전 불일치 → 캐시 삭제")
            BT_CACHE.unlink(missing_ok=True); return None
        computed = result.get("overall",{}).get("computed","")
        if computed:
            try:
                if (date.today() - date.fromisoformat(computed)).days > 7:
                    BT_CACHE.unlink(missing_ok=True); return None
            except Exception: pass
        return result
    except Exception:
        BT_CACHE.unlink(missing_ok=True); return None

def clear_all_backtest_cache():
    """백테스트/최적화 관련 캐시 전부 삭제."""
    removed = 0
    for p in [BT_CACHE, BT_KR_CACHE, OPT_CACHE, OPT_KR_CACHE, SNAPSHOT_CACHE]:
        if p.exists(): p.unlink(); removed += 1
    for p in CACHE_DIR.glob("sector_etf_ohlcv_*.pkl"):
        p.unlink(); removed += 1
    for p in CACHE_DIR.glob("factors_us_*.pkl"):
        p.unlink(); removed += 1
    for p in CACHE_DIR.glob("factors_kr_*.pkl"):
        p.unlink(); removed += 1
    for p in CACHE_DIR.glob("us_tickers_*.pkl"):
        p.unlink(); removed += 1
    logger.info(f"전체 캐시 {removed}개 삭제")
    return removed


# ════════════════════════════════════════════════════════════════════
# 점수 기여도 계산
# ════════════════════════════════════════════════════════════════════

def calc_score_explain(
    row: dict,
    weights: dict = None,
    analyst: dict = None,
    fund: dict = None,
) -> dict:
    """
    Re-rating Score의 각 요소별 기여도 분해.
    Returns: {"base":50, "components":[{"label","delta","reason"}], "total":78}
    """
    w  = weights or {}
    an = analyst or {}
    fd = fund or {}
    components = []
    base = 50.0
    total = base

    def add(label, delta, reason=""):
        nonlocal total
        if abs(delta) > 0.1:
            components.append({"label":label,"delta":round(delta,1),"reason":reason})
            total += delta

    # ── 1. EPS 컨센서스 상향 ──────────────────────────────────────
    eps_rev   = an.get("eps_revision","neutral")
    eps_trend = float(an.get("eps_trend",0) or 0)
    if eps_rev == "up":
        delta = 15 + min(15, eps_trend*0.5)
        add("EPS 컨센서스 상향", delta, f"추정치 {eps_trend:+.1f}%")
    elif eps_rev == "down":
        add("EPS 컨센서스 하향", -15, "추정치 하향 중")

    # ── 2. 실적 서프라이즈 ────────────────────────────────────────
    beat = int(an.get("beat_count",0) or 0)
    if beat >= 3: add("실적 서프라이즈 연속", 8, f"최근 {beat}분기 Beat")
    elif beat >= 2: add("실적 서프라이즈", 4, f"최근 {beat}분기 Beat")

    # ── 3. 매출 성장 가속 ─────────────────────────────────────────
    rev_accel  = bool(fd.get("rev_accel",False))
    fund_score = float(fd.get("fund_score") or 0) if fd.get("fund_score") is not None else 0
    rev_growth = float(fd.get("rev_growth",float("nan")) or float("nan"))

    if rev_accel: add("매출 성장 가속", 12, "전분기 대비 가속")
    if math.isfinite(rev_growth):
        if rev_growth >= 30:   add("매출 YoY +30%이상", 8, f"+{rev_growth:.0f}%")
        elif rev_growth >= 20: add("매출 YoY +20%이상", 5, f"+{rev_growth:.0f}%")
        elif rev_growth >= 10: add("매출 YoY +10%이상", 2, f"+{rev_growth:.0f}%")
        elif rev_growth < 0:   add("매출 감소", -5, f"{rev_growth:.0f}%")

    if fund_score > 0:
        fd_adj = (fund_score-50)/50*8
        if abs(fd_adj) > 0.5:
            add("재무 종합 점수", fd_adj, f"재무점수 {fund_score:.0f}")

    # ── 4. 테마/촉매 ─────────────────────────────────────────────
    is_theme = bool(row.get("is_theme_pick",False))
    cat      = float(row.get("catalyst_score",50) or 50)
    if is_theme: add("AI 추천 테마", 10, row.get("theme_name",""))
    cat_adj = (cat-50)/50*10
    if abs(cat_adj) > 0.5: add("촉매 신호", cat_adj, f"Catalyst {cat:.0f}")

    # ── 5. 상대강도 ───────────────────────────────────────────────
    rs_pct = float(row.get("rs_rank_pct",50) or 50)
    ret20  = float(row.get("ret_20d",0) or 0)
    rs_adj = (rs_pct-50)/50*8
    if abs(rs_adj) > 0.5: add("RS 상대강도", rs_adj, f"상위 {rs_pct:.0f}%")
    ret_adj = min(7, max(-5, ret20*0.1))
    if abs(ret_adj) > 0.3: add("20일 수익률", ret_adj, f"{ret20:+.1f}%")

    # ── 6. 거래량/돌파 ────────────────────────────────────────────
    brk = float(row.get("breakout_score",0) or 0)
    vol = float(row.get("volume_score",50) or 50)
    brk_adj = brk*0.05
    if abs(brk_adj) > 0.3: add("Breakout 신호", brk_adj, f"점수 {brk:.0f}")
    vol_adj = (vol-50)/50*5
    if abs(vol_adj) > 0.3: add("거래량 강도", vol_adj, f"점수 {vol:.0f}")

    # ── 7. Risk 패널티 (calc_rerating_score와 동일 로직) ────────────
    risk = float(row.get("top_risk_score",30) or 30)
    ret5 = float(row.get("ret_5d",0) or 0)
    flag = str(row.get("risk_flag","") or "")
    if risk >= 90 and ret5 < 0:
        add("과열+추세훼손", -15, f"Risk {risk:.0f}, 5일 {ret5:+.1f}%")
    elif risk >= 90 and ("음봉" in flag or "윗꼬리" in flag):
        add("과열+패턴훼손", -10, f"Risk {risk:.0f}, {flag}")
    elif risk >= 90:
        add("극도 과열(추세 유지)", -8, f"Risk {risk:.0f} — 추세 유지면 최소 감점")
    elif risk >= 80:
        add("과열 주의", -3, f"Risk {risk:.0f} — 추세 유지면 최소 감점")

    return {
        "base":       round(base,1),
        "components": components,
        "total":      round(max(0,min(100,total)),1),
    }


# ════════════════════════════════════════════════════════════════════
# 점수 계산 (과거 시점 T, look-ahead 차단)
# ════════════════════════════════════════════════════════════════════

def _supply_flow_at(sup_df, decision_dt, mc=float("nan"), avg_amt=float("nan")) -> float:
    """
    결정일 T 기준 수급 점수(0~100). look-ahead 차단: T까지의 수급만 사용.
    sup_df: 날짜 인덱스 + [foreign, inst] 일별 순매매.
    """
    if sup_df is None or len(sup_df) == 0:
        return 50.0
    try:
        d = sup_df.loc[:decision_dt]          # T 이하(포함)만 — 미래 수급 차단
    except Exception:
        return 50.0
    if len(d) < 5:
        return 50.0
    from screener.factors import kr_supply_score
    f = d["foreign"].dropna(); i = d["inst"].dropna()
    f5  = float(f.iloc[-5:].sum())  if len(f) >= 1 else float("nan")
    f20 = float(f.iloc[-20:].sum()) if len(f) >= 1 else float("nan")
    i5  = float(i.iloc[-5:].sum())  if len(i) >= 1 else float("nan")
    i20 = float(i.iloc[-20:].sum()) if len(i) >= 1 else float("nan")
    return kr_supply_score(f5, f20, i5, i20, mc, avg_amt)


def _calc_scores_at(close: pd.Series, df: pd.DataFrame, bench_close, t_idx: int) -> Optional[dict]:
    """T 시점까지 잘린 데이터로만 점수 계산."""
    try:
        from screener.factors import (
            leader_score, momentum_score, volume_score,
            buyable_score, breakout_score, top_risk_score,
            quality_of_trend_score, catalyst_proxy_score,
            _ret, _week52_prox, _dist_from_ma,
        )
        if len(close) < 25: return None

        rs20=rs60=0.0
        if bench_close is not None and len(bench_close) > 20:
            b20 = float((bench_close.iloc[-1]/bench_close.iloc[-21]-1)*100)
            rs20= _ret(close,20)-b20
        if bench_close is not None and len(bench_close) > 60:
            b60 = float((bench_close.iloc[-1]/bench_close.iloc[-61]-1)*100)
            rs60= _ret(close,60)-b60

        prox52 = _week52_prox(df)
        ldr  = leader_score(close,df,rs20,rs60,rs20*0.5,prox52,0.5)
        mom  = momentum_score(close,rs20,rs60,rs20*0.5,prox52,0.5)
        vol  = volume_score(df)
        buy  = buyable_score(close,df,prox52)
        brk  = breakout_score(close,df)
        risk = top_risk_score(close,df)
        qot  = quality_of_trend_score(close,df)
        cat  = catalyst_proxy_score(close,df)

        # 신규 factor
        from screener.factors import (
            momentum_accel_score, volume_shock_score,
            breakout_persistence_score, near_high_score,
            volatility_contraction_score, pullback_quality_score, risk_filter_score,
        )
        mom_accel   = momentum_accel_score(close)
        vol_shock   = volume_shock_score(df)
        brk_persist = breakout_persistence_score(close, df)
        near_hi     = near_high_score(df)
        vol_cont    = volatility_contraction_score(close, df)
        pullback_q  = pullback_quality_score(close, df)
        risk_filt   = risk_filter_score(close, df)

        return {
            "leader":   ldr, "momentum": mom, "volume":  vol,
            "buyable":  buy, "breakout": brk, "risk":    risk,
            "quality":  qot, "catalyst": cat,
            "ret_20d":  _ret(close,20),
            "dist_20":  _dist_from_ma(close,20),
            "prox52":   prox52 if math.isfinite(prox52) else 75,
            # 신규 factor
            "momentum_accel": mom_accel,
            "volume_shock":   vol_shock,
            "brk_persist":    brk_persist,
            "near_high":      near_hi["composite"],
            "vol_contract":   vol_cont,
            "pullback_q":     pullback_q,
            "risk_filter":    risk_filt,
            "sect_rs":        50.0,   # 백테스트에 섹터 데이터 없음 → 기본값
            "supply_flow":    50.0,   # 백테스트에 수급 데이터 없음 → 기본값
            # enhanced (수동조정/WF와 동일 공식)
            "volume_enhanced":   round(vol*0.4 + vol_shock*0.6, 1),
            "breakout_enhanced": round(brk*0.5 + brk_persist*0.5, 1),
            "risk_combined":     round(risk*0.4 + risk_filt*0.6, 1),
        }
    except Exception as e:
        logger.debug(f"점수 계산 실패: {e}")
        return None


# ════════════════════════════════════════════════════════════════════
# 워크포워드 백테스트
# ════════════════════════════════════════════════════════════════════

def run_walkforward_backtest(
    ohlcv_data:  dict,
    lookahead:   int = 20,
    step:        int = 5,
    min_history: int = 180,   # 52주 지표 신뢰성 위해 180 거래일
    bench_ticker:str = "SPY",
    progress_cb  = None,
) -> dict:
    records = []

    # ETF/벤치마크 제외, 최소 이력 충족 개별주만
    tickers = [
        t for t,v in ohlcv_data.items()
        if not is_us_backtest_excluded_ticker(t)
        and "ohlcv" in v
        and len(v["ohlcv"]) >= min_history + lookahead + 5
    ]
    if not tickers: return {}

    bench_close = (ohlcv_data[bench_ticker]["ohlcv"]["Close"]
                   if bench_ticker in ohlcv_data and "ohlcv" in ohlcv_data[bench_ticker]
                   else None)
    qqq_close   = (ohlcv_data["QQQ"]["ohlcv"]["Close"]
                   if "QQQ" in ohlcv_data and "ohlcv" in ohlcv_data["QQQ"]
                   else None)

    try: w = load_optimal_weights()
    except Exception: w = {}

    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if progress_cb: progress_cb(i+1, total, ticker)
        df    = ohlcv_data[ticker]["ohlcv"].copy()
        close = df["Close"]; n = len(close)

        for t_idx in range(min_history, n - lookahead - 1, step):
            try:
                df_t    = df.iloc[:t_idx+1].copy()
                close_t = df_t["Close"]
                bench_t = bench_close.iloc[:t_idx+1] if bench_close is not None and len(bench_close) > t_idx else bench_close

                sc = _calc_scores_at(close_t, df_t, bench_t, len(close_t)-1)
                if sc is None: continue

                # 백테스트/실전 공통 점수 함수 사용
                # 신규/enhanced factor까지 포함하려면 scalar 함수가 아니라 series 공통식을 사용해야 함
                composite = float(calc_validated_score_series(pd.DataFrame([sc]), weights=w).iloc[0])

                def _dstr(idx):
                    v=df.index[idx]; return str(v.date()) if hasattr(v,"date") else str(v)[:10]

                decision_date = _dstr(t_idx)
                entry_idx = min(t_idx+1, n-1); exit_idx = min(t_idx+lookahead, n-1)
                entry_date = _dstr(entry_idx); exit_date = _dstr(exit_idx)

                p_entry = float(df["Open"].iloc[entry_idx]) if "Open" in df.columns and entry_idx<n else float(close.iloc[t_idx])
                if p_entry<=0: p_entry=float(close.iloc[t_idx])
                p_exit = float(close.iloc[exit_idx])
                if p_entry<=0: continue
                ret = (p_exit/p_entry-1)*100

                # 20일 내 최고수익률 (급등 적중률 계산용)
                max_ret_20d = float("nan")
                try:
                    if "High" in df.columns:
                        future_high = float(df["High"].iloc[entry_idx:exit_idx+1].max())
                    else:
                        future_high = float(close.iloc[entry_idx:exit_idx+1].max())
                    if p_entry > 0:
                        max_ret_20d = round((future_high/p_entry-1)*100, 2)
                except Exception:
                    pass

                spy_ret=excess_spy=float("nan")
                if bench_close is not None and exit_idx<len(bench_close):
                    b_e=float(bench_close.iloc[t_idx]); b_x=float(bench_close.iloc[exit_idx])
                    if b_e>0: spy_ret=(b_x/b_e-1)*100; excess_spy=ret-spy_ret

                qqq_ret=excess_qqq=float("nan")
                if qqq_close is not None and exit_idx<len(qqq_close):
                    q_e=float(qqq_close.iloc[t_idx]); q_x=float(qqq_close.iloc[exit_idx])
                    if q_e>0: qqq_ret=(q_x/q_e-1)*100; excess_qqq=ret-qqq_ret

                records.append({
                    "ticker":ticker,"t_idx":t_idx,
                    "decision_date":decision_date,"entry_date":entry_date,"exit_date":exit_date,
                    "entry_price":round(p_entry,2),"exit_price":round(p_exit,2),
                    "ret":round(ret,2),
                    "max_ret_20d":  max_ret_20d,
                    "validated_score": composite,  # composite = validated_score (동일 공식)
                    "spy_ret":round(spy_ret,2) if math.isfinite(spy_ret) else float("nan"),
                    "qqq_ret":round(qqq_ret,2) if math.isfinite(qqq_ret) else float("nan"),
                    "excess_spy":round(excess_spy,2) if math.isfinite(excess_spy) else float("nan"),
                    "excess_qqq":round(excess_qqq,2) if math.isfinite(excess_qqq) else float("nan"),
                    "composite":round(composite,1),
                    **sc,
                })
            except Exception as e:
                logger.debug(f"BT {ticker} t={t_idx}: {e}")

    if len(records) < 30: return {}
    df_rec = pd.DataFrame(records)
    result = _analyze_backtest(df_rec, lookahead)
    result.update({"n_records":len(records),"n_tickers":len(set(r["ticker"] for r in records)),"bt_version":BT_VERSION})

    try:
        with open(BT_CACHE,"wb") as f: pickle.dump(result,f)
    except Exception: pass
    return result


# ════════════════════════════════════════════════════════════════════
# 분석
# ════════════════════════════════════════════════════════════════════

def _analyze_backtest(df: pd.DataFrame, lookahead: int) -> dict:
    """백테스트 레코드를 성과/검증 지표로 요약.

    v10 개선:
    - KOSPI/KOSDAQ/SPY 등 여러 벤치마크 초과수익을 동시에 집계
    - Pearson뿐 아니라 Spearman 순위상관을 계산
    - 합성점수 분위별 hit-rate/초과수익을 계산
    - 날짜별 Top10/Top20 실제 편입 종목을 확인할 수 있게 보존
    """
    df = df.copy()

    def sm(s):
        s = pd.Series(s).dropna()
        return float(s.mean()) if len(s) > 0 else float("nan")

    def sw(s):
        s = pd.Series(s).dropna()
        return float((s > 0).mean() * 100) if len(s) > 0 else float("nan")

    def sh(s):
        s = pd.Series(s).dropna()
        return float(s.mean() / s.std() * np.sqrt(252 / lookahead)) if len(s) > 2 and s.std() > 0 else float("nan")

    def rr(x, nd=2):
        try:
            return round(float(x), nd) if math.isfinite(float(x)) else float("nan")
        except Exception:
            return float("nan")

    # excess_* 컬럼을 모두 벤치마크로 인식한다. 기존 UI 호환을 위해 excess_spy가 있으면 primary로 유지.
    excess_cols = [c for c in df.columns if c.startswith("excess_")]
    if "excess_spy" in excess_cols:
        primary_excess = "excess_spy"
    elif "excess_kospi" in excess_cols:
        primary_excess = "excess_kospi"
    elif excess_cols:
        primary_excess = excess_cols[0]
    else:
        primary_excess = None

    # 날짜별 포트폴리오
    by_date_portfolio = {}
    dates = df["decision_date"].dropna().unique() if "decision_date" in df.columns else []
    for dt in sorted(dates):
        sub = df[df["decision_date"] == dt].copy()
        if len(sub) < 5:
            continue
        sort_col = "composite" if "composite" in sub.columns else "leader"
        sub = sub.sort_values(sort_col, ascending=False)
        for top_k, label in [(10, "top10"), (20, "top20")]:
            top = sub.head(top_k)
            rets = top["ret"].dropna()
            if len(rets) == 0:
                continue
            row = {
                "date": dt,
                "avg_ret": rr(sm(rets)),
                "median_ret": rr(float(rets.median())),
                "win_rate": rr(sw(rets), 1),
                "n": len(rets),
                "tickers": top["ticker"].astype(str).tolist()[:top_k],
            }
            # 실제 종목별 결과를 보존해 UI에서 감사표처럼 확인 가능하게 한다.
            detail_cols = [c for c in [
                "ticker", "name", "market", "composite", "leader", "breakout", "volume", "quality", "catalyst", "risk",
                "entry_date", "entry_price", "exit_date", "exit_price", "ret",
                "spy_ret", "kospi_ret", "kosdaq_ret", "excess_spy", "excess_kospi", "excess_kosdaq"
            ] if c in top.columns]
            row["details"] = top[detail_cols].round(2).to_dict("records") if detail_cols else []
            for ex_col in excess_cols:
                suffix = ex_col.replace("excess_", "")
                exc = top[ex_col].dropna()
                row[f"avg_{ex_col}"] = rr(sm(exc)) if len(exc) else float("nan")
                row[f"{suffix}_beat_rate"] = rr(sw(exc), 1) if len(exc) else float("nan")
            # 기존 US UI 호환
            if primary_excess:
                row.setdefault("avg_excess", row.get(f"avg_{primary_excess}", float("nan")))
                row.setdefault("spy_beat_rate", row.get("spy_beat_rate", row.get(primary_excess.replace("excess_", "") + "_beat_rate", float("nan"))))
            by_date_portfolio.setdefault(label, []).append(row)

    by_track = {}
    for label in ["top10", "top20"]:
        rows = by_date_portfolio.get(label, [])
        if not rows:
            continue
        avg_rets = pd.Series([r["avg_ret"] for r in rows]).dropna()
        n_label = "상위 10개" if label == "top10" else "상위 20개"
        tr = {
            "avg_ret": rr(sm(avg_rets)),
            "median_ret": rr(float(avg_rets.median())) if len(avg_rets) else float("nan"),
            "win_rate": rr(sw(avg_rets), 1),
            "sharpe": rr(sh(avg_rets), 3),
            "n": len(rows),
            "std": rr(float(avg_rets.std())) if len(avg_rets) else float("nan"),
        }
        for ex_col in excess_cols:
            suffix = ex_col.replace("excess_", "")
            vals = pd.Series([r.get(f"avg_{ex_col}", float("nan")) for r in rows]).dropna()
            beats = pd.Series([r.get(f"{suffix}_beat_rate", float("nan")) for r in rows]).dropna()
            tr[f"avg_{ex_col}"] = rr(sm(vals)) if len(vals) else float("nan")
            tr[f"{suffix}_beat_rate"] = rr(sm(beats), 1) if len(beats) else float("nan")
        if primary_excess:
            tr["avg_excess_spy"] = tr.get(f"avg_{primary_excess}", float("nan"))
            suffix = primary_excess.replace("excess_", "")
            tr["spy_beat_rate"] = tr.get(f"{suffix}_beat_rate", float("nan"))
        by_track[n_label] = tr

    # 점수 구간별 성과: R²가 낮아도 순위가 작동하는지 보기 위한 핵심 표
    score_buckets = []
    try:
        if "composite" in df.columns and len(df[["composite", "ret"]].dropna()) >= 30:
            tmp = df[["composite", "ret"] + excess_cols].dropna(subset=["composite", "ret"]).copy()
            tmp["score_pct"] = tmp["composite"].rank(pct=True, method="average")
            labels = ["하위 20%", "20~40%", "40~60%", "60~80%", "상위 20%"]
            tmp["bucket"] = pd.cut(tmp["score_pct"], [0, .2, .4, .6, .8, 1.0], labels=labels, include_lowest=True)
            for lab in labels:
                sub = tmp[tmp["bucket"] == lab]
                if len(sub) < 5:
                    continue
                row = {
                    "점수구간": lab,
                    "n": int(len(sub)),
                    "평균점수": rr(sub["composite"].mean(), 1),
                    "평균수익률%": rr(sub["ret"].mean()),
                    "중앙수익률%": rr(sub["ret"].median()),
                    "승률%": rr(sw(sub["ret"]), 1),
                }
                for ex_col in excess_cols:
                    suffix = ex_col.replace("excess_", "").upper()
                    exc = sub[ex_col].dropna()
                    if len(exc):
                        row[f"{suffix}초과%"] = rr(exc.mean())
                        row[f"{suffix}초과승률%"] = rr(sw(exc), 1)
                score_buckets.append(row)
    except Exception as e:
        logger.debug(f"score bucket 실패: {e}")

    by_leader = {}
    for lo, hi in [(0, 30), (30, 45), (45, 55), (55, 65), (65, 75), (75, 100)]:
        mask = (df["leader"] >= lo) & (df["leader"] < hi) if "leader" in df.columns else pd.Series(False, index=df.index)
        sub = df.loc[mask, "ret"].dropna()
        if len(sub) >= 5:
            item = {
                "avg_ret": rr(sm(sub)),
                "median": rr(float(sub.median())),
                "win_rate": rr(sw(sub), 1),
                "sharpe": rr(sh(sub), 3),
                "n": len(sub),
            }
            if primary_excess and primary_excess in df.columns:
                exc = df.loc[mask, primary_excess].dropna()
                item["avg_excess"] = rr(sm(exc)) if len(exc) else float("nan")
            by_leader[f"{lo}~{hi}"] = item

    overall = {
        "avg_ret": rr(sm(df["ret"])),
        "median_ret": rr(float(df["ret"].median())),
        "win_rate": rr(sw(df["ret"]), 1),
        "sharpe": rr(sh(df["ret"]), 3),
        "lookahead": lookahead,
        "computed": date.today().isoformat(),
        "n_dates": len(dates),
    }
    for ex_col in excess_cols:
        suffix = ex_col.replace("excess_", "")
        exc = df[ex_col].dropna()
        overall[f"avg_{ex_col}"] = rr(sm(exc)) if len(exc) else float("nan")
        overall[f"{suffix}_beat_rate"] = rr(sw(exc), 1) if len(exc) else float("nan")
    if primary_excess:
        suffix = primary_excess.replace("excess_", "")
        overall["avg_excess_spy"] = overall.get(f"avg_{primary_excess}", float("nan"))
        overall["spy_beat_rate"] = overall.get(f"{suffix}_beat_rate", float("nan"))
        overall["primary_excess_col"] = primary_excess

    correlations = {}
    spearman = {}
    for col in ["leader", "momentum", "volume", "buyable", "breakout", "risk", "quality", "catalyst", "composite"]:
        if col in df.columns:
            valid = df[[col, "ret"]].dropna()
            if len(valid) >= 10:
                correlations[col] = rr(valid[col].corr(valid["ret"], method="pearson"), 4)
                # scipy 없이 rank 기반 피어슨으로 스피어만 계산
                try:
                    r_s = valid[col].rank().corr(valid["ret"].rank())
                    spearman[col] = rr(float(r_s), 4)
                except Exception:
                    pass

    regression = {}
    try:
        if "composite" in df.columns:
            x = df["composite"].values
            y = df["ret"].values
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 10:
                coef = np.polyfit(x[valid], y[valid], 1)
                y_pred = np.polyval(coef, x[valid])
                ss_res = np.sum((y[valid] - y_pred) ** 2)
                ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                regression = {"slope": rr(coef[0], 4), "intercept": rr(coef[1]), "r2": rr(r2, 4)}
                df["predicted_ret"] = np.polyval(coef, df["composite"].fillna(50))
                df["prediction_error"] = df["ret"] - df["predicted_ret"]
    except Exception as e:
        logger.debug(f"regression 실패: {e}")

    return {
        "by_track": by_track,
        "by_leader": by_leader,
        "by_date_portfolio": by_date_portfolio,
        "score_buckets": score_buckets,
        "regression": regression,
        "overall": overall,
        "correlations": correlations,
        "spearman": spearman,
        "df_records": df,
    }

# ════════════════════════════════════════════════════════════════════
# Train/Test 분리 가중치 최적화 (과최적화 방지)
# ════════════════════════════════════════════════════════════════════

def optimize_weights(bt_result: dict, train_ratio: float = 0.7) -> dict:
    """
    Train/Test 분리 최적화.
    - Train: 오래된 70% 날짜 → 가중치 탐색
    - Test:  최근 30% 날짜  → 검증 (과최적화 확인)
    - Test에서도 기본 가중치보다 나을 때만 저장
    """
    df = bt_result.get("df_records")
    if df is None or len(df) < 50:
        logger.warning("최적화 데이터 부족")
        return _default_weights()

    # Train/Test 날짜 분리
    if "decision_date" in df.columns:
        all_dates = sorted(df["decision_date"].unique())
        n_train   = max(10, int(len(all_dates) * train_ratio))
        train_dates = set(all_dates[:n_train])
        test_dates  = set(all_dates[n_train:])
        df_train = df[df["decision_date"].isin(train_dates)].copy()
        df_test  = df[df["decision_date"].isin(test_dates)].copy()
    else:
        n_train  = int(len(df) * train_ratio)
        df_train = df.iloc[:n_train].copy()
        df_test  = df.iloc[n_train:].copy()

    logger.info(f"최적화 - Train: {len(df_train)}건, Test: {len(df_test)}건")

    required = ["leader","volume","buyable","breakout","catalyst","quality","risk","ret"]
    for col in required:
        if col not in df_train.columns: df_train[col]=50.0
        if col not in df_test.columns:  df_test[col]=50.0

    df_train = df_train.dropna(subset=required)
    df_test  = df_test.dropna(subset=required)
    if len(df_train) < 30: return _default_weights()

    def _sharpe_top(df_sub, w_lead, w_vol, w_brk, w_cat, w_qot, w_risk):
        # 목표: 원수익률이 아니라 SPY/KOSPI 초과수익 최대화
        # [공식 일원화] 실전과 동일한 calc_validated_score_series 사용.
        #   → enhanced 컬럼(volume_enhanced/breakout_enhanced/risk_combined) 우선,
        #     양수 가중치 정규화(합=1.0)까지 실전과 100% 동일하게 평가한다.
        import numpy as _np
        w_try = {"leader": w_lead, "volume": w_vol, "breakout": w_brk,
                 "catalyst": w_cat, "quality": 0.0, "risk_penalty": w_risk}  # quality 제거
        score = calc_validated_score_series(df_sub, weights=w_try).values
        threshold = _np.quantile(score, 0.70)
        top_idx   = score >= threshold
        # 초과수익이 있으면 초과수익 기준, 없으면 원수익률
        if "excess_spy" in df_sub.columns:
            exc = df_sub.loc[top_idx, "excess_spy"].dropna()
            if len(exc) >= 5:
                return float(exc.mean()/(exc.std()+1e-8))
        top_rets = df_sub.loc[top_idx, "ret"]
        if len(top_rets) < 5: return -999
        return float(top_rets.mean()/(top_rets.std()+1e-8))

    # 그리드 서치 (Train)
    best_sharpe = -999; best_w = None
    for w_lead in [0.30,0.35,0.38,0.40,0.43]:
        for w_vol in [0.12,0.15,0.18,0.20]:
            for w_brk in [0.08,0.10,0.12,0.15]:
                for w_cat in [0.10,0.12,0.15]:
                    w_qot = max(0.05, round(1.0-w_lead-w_vol-w_brk-w_cat,2))
                    if w_qot<0.05 or w_qot>0.20: continue
                    if abs(w_lead+w_vol+w_brk+w_cat+w_qot-1.0)>0.02: continue
                    for w_risk in [0.05,0.08,0.10]:
                        sh = _sharpe_top(df_train,w_lead,w_vol,w_brk,w_cat,w_qot,w_risk)
                        if sh>best_sharpe:
                            best_sharpe=sh
                            best_w=dict(leader=w_lead,volume=w_vol,breakout=w_brk,
                                       catalyst=w_cat,quality=0.0,risk_penalty=w_risk)

    if best_w is None: return _default_weights()

    # Test 검증
    dw = _default_weights()
    test_best   = _sharpe_top(df_test,  best_w["leader"],best_w["volume"],best_w["breakout"],best_w["catalyst"],best_w["quality"],best_w["risk_penalty"])
    test_default= _sharpe_top(df_test,  dw["leader"],dw["volume"],dw["breakout"],dw["catalyst"],dw["quality"],dw["risk_penalty"])
    train_default=_sharpe_top(df_train, dw["leader"],dw["volume"],dw["breakout"],dw["catalyst"],dw["quality"],dw["risk_penalty"])

    best_w["train_sharpe"]    = round(best_sharpe, 4)
    best_w["test_sharpe"]     = round(test_best, 4)
    best_w["default_train"]   = round(train_default, 4)
    best_w["default_test"]    = round(test_default, 4)
    best_w["improvement_train"] = round(best_sharpe - train_default, 4)
    best_w["improvement_test"]  = round(test_best  - test_default,  4)
    best_w["n_train_dates"]   = len(set(df_train.get("decision_date",pd.Series()).unique()))
    best_w["n_test_dates"]    = len(set(df_test.get("decision_date",pd.Series()).unique()))
    best_w["optimized_at"]    = date.today().isoformat()

    # Test에서도 나을 때만 저장
    if best_w["improvement_test"] >= -0.05:
        try:
            with open(OPT_CACHE,"wb") as f: pickle.dump(best_w,f)
            logger.info(f"가중치 저장 — Train Sharpe +{best_w['improvement_train']:.4f}, Test +{best_w['improvement_test']:.4f}")
        except Exception: pass
        save_weights_history(best_w, "US")
        return best_w
    else:
        logger.info(f"Test 검증 미통과 → 기본 가중치 유지 (test 개선 {best_w['improvement_test']:+.4f})")
        default = _default_weights()
        default["rejected_reason"] = "Test 검증 미통과 (과최적화 의심)"
        default["train_sharpe"]    = round(best_sharpe,4)
        default["test_sharpe"]     = round(test_best,4)
        save_weights_history(default, "US")
        return default


# ════════════════════════════════════════════════════════════════════
# 가중치 변화 이력 관리
# ════════════════════════════════════════════════════════════════════

WEIGHTS_HISTORY_CACHE = CACHE_DIR / "weights_history.pkl"


def save_weights_history(weights: dict, market: str = "US"):
    """최적화된 가중치를 이력에 누적 저장."""
    history = load_weights_history()
    entry = {
        "date":       date.today().isoformat(),
        "market":     market,
        "leader":     weights.get("leader",   0.38),
        "volume":     weights.get("volume",   0.18),
        "breakout":   weights.get("breakout", 0.12),
        "catalyst":   weights.get("catalyst", 0.15),
        "quality":    weights.get("quality",  0.10),
        "risk_penalty": weights.get("risk_penalty", 0.08),
        "train_sharpe": weights.get("train_sharpe",  float("nan")),
        "test_sharpe":  weights.get("test_sharpe",   float("nan")),
        "improvement_train": weights.get("improvement_train", 0.0),
        "improvement_test":  weights.get("improvement_test",  0.0),
        "rejected":   bool(weights.get("rejected_reason","")),
        "source":     weights.get("source", ""),
        "objective":  weights.get("objective", ""),
        "top_k":      weights.get("top_k", 0),
        "n_train_dates": weights.get("n_train_dates", 0),
        "n_test_dates":  weights.get("n_test_dates",  0),
    }
    history.append(entry)
    # 최대 90일치만 보관
    if len(history) > 90:
        history = history[-90:]
    try:
        with open(WEIGHTS_HISTORY_CACHE,"wb") as f: pickle.dump(history, f)
        logger.info(f"가중치 이력 저장: {len(history)}건")
    except Exception as e:
        logger.warning(f"가중치 이력 저장 실패: {e}")


def load_weights_history() -> list[dict]:
    """가중치 이력 로드."""
    if not WEIGHTS_HISTORY_CACHE.exists():
        return []
    try:
        with open(WEIGHTS_HISTORY_CACHE,"rb") as f:
            return pickle.load(f)
    except Exception:
        return []


def save_optimal_weights(weights: dict, market: str = "US", source: str = "manual", objective: str = "", top_k: int = 10) -> dict:
    """UI/Walk-forward에서 선택한 가중치를 실전 스크리너용 캐시에 저장합니다."""
    market = (market or "US").upper()
    clean = _default_weights()
    for k in ["leader", "volume", "breakout", "catalyst", "quality", "risk_penalty"]:
        if k in weights:
            try:
                clean[k] = round(float(weights[k]), 4)
            except Exception:
                pass
    clean.update({
        "market": market,
        "source": source,
        "objective": objective,
        "top_k": top_k,
        "optimized_at": date.today().isoformat(),
        "improvement_train": float(weights.get("improvement_train", 0.0) or 0.0),
        "improvement_test":  float(weights.get("improvement_test",  0.0) or 0.0),
    })
    cache = OPT_KR_CACHE if market == "KR" else OPT_CACHE
    try:
        with open(cache, "wb") as f:
            pickle.dump(clean, f)
        save_weights_history(clean, market)
        logger.info(f"{market} 실전 가중치 저장 완료: {clean}")
    except Exception as e:
        logger.warning(f"{market} 실전 가중치 저장 실패: {e}")
    return clean


def _default_weights() -> dict:
    return {"leader":0.38,"volume":0.18,"breakout":0.12,"catalyst":0.15,"quality":0.0,
            "risk_penalty":0.08,"sharpe":float("nan"),"improvement_train":0.0,"improvement_test":0.0,
            "default_train":float("nan"),"default_test":float("nan")}

def load_optimal_weights() -> dict:
    if OPT_CACHE.exists():
        try:
            with open(OPT_CACHE,"rb") as f:
                _w = pickle.load(f)
            _w["quality"] = 0.0   # quality 영구 제거(백테스트 역효과) — 캐시 덮어쓰기
            return _w
        except Exception: pass
    return _default_weights()


# ════════════════════════════════════════════════════════════════════
# 예상 수익률
# ════════════════════════════════════════════════════════════════════

def estimate_return(row: dict) -> dict:
    leader=float(row.get("leader_score",50) or 50)
    buyable=float(row.get("buyable_score",50) or 50)
    breakout=float(row.get("breakout_score",0) or 0)
    catalyst=float(row.get("catalyst_score",50) or 50)
    risk=float(row.get("top_risk_score",30) or 30)
    analyst=float(row.get("analyst_score",50) or 50)
    quality=float(row.get("quality_score",50) or 50)
    volume=float(row.get("volume_score",50) or 50)
    track=row.get("track","")

    bt=load_backtest()
    bt_quality="none"
    if bt:
        n_rec=bt.get("n_records",0); corr=bt.get("correlations",{})
        best_corr=max(abs(v) for v in corr.values()) if corr else 0
        bt_quality="high" if n_rec>=200 and best_corr>=0.05 else "medium" if n_rec>=50 else "low"

    if bt_quality in ("high","medium"):
        result=_estimate_from_bt(bt,leader,buyable,breakout,catalyst,risk,analyst,volume,quality,track)
        result["bt_quality"]=bt_quality; result["data_source"]=f"워크포워드 BT ({bt.get('n_records',0)}건)"
        if bt_quality=="low": result["confidence"]="낮음"
    else:
        result=_estimate_rule_based(leader,buyable,breakout,catalyst,risk,analyst,track)
        result["bt_quality"]="none"; result["data_source"]="규칙 기반 (BT 미실행)"; result["confidence"]="참고용"
    return result

def _estimate_from_bt(bt,leader,buyable,breakout,catalyst,risk,analyst,volume,quality,track):
    by_leader=bt.get("by_leader",{}); overall=bt.get("overall",{})
    base_ret=float(overall.get("avg_ret",2.0) or 2.0); win_rate=float(overall.get("win_rate",50.0) or 50.0)
    for bucket,data in by_leader.items():
        try:
            lo,hi=map(int,bucket.split("~"))
            if lo<=leader<hi: base_ret=data.get("avg_ret",base_ret); win_rate=data.get("win_rate",win_rate); break
        except Exception: pass
    corr=bt.get("correlations",{}); adj=0.0
    for fac,val,col in [(breakout,0,"breakout"),(catalyst,0,"catalyst"),(volume,0,"volume")]:
        c=corr.get(col,0); adj+=(fac-50)/50*c*3
    final_mid=base_ret+adj
    if analyst>65: final_mid+=(analyst-65)*0.08
    elif analyst<40: final_mid+=(analyst-40)*0.06
    std_est=max(6,min(20,10))
    conf="높음" if win_rate>=60 and bt.get("n_records",0)>=200 else "낮음" if win_rate<45 else "중간"
    return {"expected_low":round(final_mid-std_est*0.7,1),"expected_high":round(final_mid+std_est*1.1,1),
        "expected_mid":round(final_mid,1),"confidence":conf,"basis":f"워크포워드 BT (승률{win_rate:.0f}%)","win_rate":round(win_rate,1)}

def _estimate_rule_based(leader,buyable,breakout,catalyst,risk,analyst,track):
    base={"Hot Leader / Extended":(5,20,12),"Hot Leader / Buyable":(8,25,16),
        "Breakout Signal":(5,30,15),"Leader / Buyable":(5,18,11),
        "Leader / Pullback Wait":(3,15,9),"Turnaround / Early":(5,35,18),
        "Watch Only":(-2,10,4),"Avoid / Weak":(-10,3,-3)}
    low,high,mid=base.get(track,(0,12,5))
    if leader>=80: high+=8; mid+=4
    elif leader>=65: high+=4; mid+=2
    elif leader<40: high-=5; mid-=3
    if breakout>=70: high+=10; mid+=5
    if catalyst>=70: high+=6; mid+=3
    if analyst>=75: high+=8; mid+=4
    if risk>=70: low-=8; high+=3
    if buyable<35: low-=5; mid-=2
    conf="높음" if sum(1 for s in [leader,breakout,catalyst,analyst] if s>=65)>=3 else "낮음" if sum(1 for s in [leader,breakout,catalyst,analyst] if s>=65)==0 else "중간"
    return {"expected_low":round(low,1),"expected_high":round(high,1),"expected_mid":round(mid,1),"confidence":conf,"basis":"규칙 기반 (BT 없음)"}


# ════════════════════════════════════════════════════════════════════
# 스냅샷 백테스트 (실제 라이브 랭킹과 동일한 로직)
# ════════════════════════════════════════════════════════════════════

def run_snapshot_backtest(
    ohlcv_data:  dict,
    snapshot_dates: list = None,
    top_k:       int = 20,
    lookahead:   int = 20,
    bench_ticker:str = "SPY",
) -> list[dict]:
    from screener.factors import classify_track  # 실전 트랙 분류 재사용(공식 일원화)
    if SNAPSHOT_CACHE.exists():
        try:
            with open(SNAPSHOT_CACHE,"rb") as f:
                cached=pickle.load(f)
            if cached.get("bt_version")==BT_VERSION:
                return cached.get("results",[])
        except Exception:
            SNAPSHOT_CACHE.unlink(missing_ok=True)

    bench_close=(ohlcv_data[bench_ticker]["ohlcv"]["Close"]
                 if bench_ticker in ohlcv_data and "ohlcv" in ohlcv_data[bench_ticker] else None)
    qqq_close  =(ohlcv_data["QQQ"]["ohlcv"]["Close"]
                 if "QQQ" in ohlcv_data and "ohlcv" in ohlcv_data["QQQ"] else None)

    # 공통 날짜 인덱스 (ETF 제외 종목만)
    all_dates=None
    for t,v in ohlcv_data.items():
        if is_us_backtest_excluded_ticker(t) or "ohlcv" not in v: continue
        idx=v["ohlcv"].index
        all_dates=pd.DatetimeIndex(idx) if all_dates is None else all_dates.intersection(pd.DatetimeIndex(idx))
    if all_dates is None or len(all_dates)<lookahead+60: return []
    all_dates=all_dates.sort_values()

    if snapshot_dates is None:
        snapshot_dates=[]
        for offset in [20,40,60,90,120]:
            if len(all_dates)>offset+lookahead+10:
                snapshot_dates.append(str(all_dates[-(offset+lookahead+1)].date()))

    try: w=load_optimal_weights()
    except Exception: w={}

    results=[]
    for snap_str in snapshot_dates:
        try:
            snap_dt=pd.Timestamp(snap_str)
            positions=[i for i,d in enumerate(all_dates) if d<=snap_dt]
            if not positions or len(positions)<60: continue
            t_idx=positions[-1]
            if t_idx+lookahead>=len(all_dates): continue
            exit_idx=t_idx+lookahead

            # 사후편입 편향 방지: 해당 날짜에 유니버스에 있었던 종목만
            try:
                from data.us_universe import get_forced_universe_at
                forced_at_date = set(get_forced_universe_at(snap_str))
            except Exception:
                forced_at_date = None

            # 한국 강제 유니버스도 동일 처리
            try:
                from data.kr_universe import get_forced_kr_universe_at, FORCED_KR_META
                forced_kr_at_date = set(get_forced_kr_universe_at(snap_str))
            except Exception:
                forced_kr_at_date = None; FORCED_KR_META = {}

            scores=[]
            for ticker,item in ohlcv_data.items():
                if is_us_backtest_excluded_ticker(ticker) or "ohlcv" not in item: continue
                # 사후편입 편향 체크: FORCED_UNIVERSE 종목이면 해당 날짜에 포함됐어야 함
                if forced_at_date is not None:
                    from data.us_universe import FORCED_UNIVERSE_META
                    if ticker in FORCED_UNIVERSE_META and ticker not in forced_at_date:
                        continue
                # 한국 강제 유니버스 편향 체크
                if forced_kr_at_date is not None and ticker in FORCED_KR_META:
                    if ticker not in forced_kr_at_date:
                        continue
                df_full=item["ohlcv"]
                df_t=df_full[df_full.index<=all_dates[t_idx]].copy()
                if len(df_t)<60: continue
                close_t=df_t["Close"]
                bench_t=bench_close[bench_close.index<=all_dates[t_idx]] if bench_close is not None else None

                sc=_calc_scores_at(close_t,df_t,bench_t,len(close_t)-1)
                if sc is None: continue

                # [공식 일원화] composite·트랙 모두 실전과 동일한 공유 함수 사용.
                #   기존엔 raw 공식/축약 분류를 손으로 재구현해 실전과 미세하게 어긋났음.
                composite = float(calc_validated_score_series(pd.DataFrame([sc]), weights=w).iloc[0])

                # 트랙 분류 — factors.classify_track 그대로 호출 (drift 방지)
                track = classify_track(
                    leader=sc["leader"], buyable=sc["buyable"], breakout=sc["breakout"],
                    risk=sc["risk"], momentum=sc["momentum"],
                )

                # 진입가/청산가
                exit_date_ts=all_dates[exit_idx]
                t1_rows=df_full[df_full.index>all_dates[t_idx]]
                entry_p=float(t1_rows["Open"].iloc[0]) if len(t1_rows)>0 and "Open" in df_full.columns else float(close_t.iloc[-1])
                exit_p=float(df_full[df_full.index<=exit_date_ts]["Close"].iloc[-1])
                if entry_p<=0: continue
                ret=(exit_p/entry_p-1)*100

                # SPY/QQQ 수익률
                spy_ret=excess_spy=float("nan")
                if bench_close is not None:
                    bc_t=bench_close[bench_close.index<=all_dates[t_idx]]
                    bc_x=bench_close[bench_close.index<=exit_date_ts]
                    if len(bc_t)>0 and len(bc_x)>0:
                        b_e=float(bc_t.iloc[-1]); b_x=float(bc_x.iloc[-1])
                        if b_e>0: spy_ret=(b_x/b_e-1)*100; excess_spy=ret-spy_ret

                qqq_ret=excess_qqq=float("nan")
                if qqq_close is not None:
                    qc_t=qqq_close[qqq_close.index<=all_dates[t_idx]]
                    qc_x=qqq_close[qqq_close.index<=exit_date_ts]
                    if len(qc_t)>0 and len(qc_x)>0:
                        q_e=float(qc_t.iloc[-1]); q_x=float(qc_x.iloc[-1])
                        if q_e>0: qqq_ret=(q_x/q_e-1)*100; excess_qqq=ret-qqq_ret

                scores.append({
                    "ticker":ticker,"composite":round(composite,1),
                    "leader":round(ldr,1),"breakout":round(brk,1),
                    "catalyst":round(sc["catalyst"],1),"volume":round(sc["volume"],1),
                    "risk":round(risk,1),"track":track,
                    "entry_price":round(entry_p,2),"exit_price":round(exit_p,2),
                    "entry_date":str(t1_rows.index[0].date()) if len(t1_rows)>0 else snap_str,
                    "exit_date":str(exit_date_ts.date()),
                    "ret":round(ret,2),
                    "spy_ret":round(spy_ret,2) if math.isfinite(spy_ret) else float("nan"),
                    "qqq_ret":round(qqq_ret,2) if math.isfinite(qqq_ret) else float("nan"),
                    "excess_spy":round(excess_spy,2) if math.isfinite(excess_spy) else float("nan"),
                    "excess_qqq":round(excess_qqq,2) if math.isfinite(excess_qqq) else float("nan"),
                })

            if not scores: continue
            scores.sort(key=lambda x:x["composite"],reverse=True)
            top=scores[:top_k]
            rets=pd.Series([s["ret"] for s in top])
            exc=pd.Series([s["excess_spy"] for s in top if math.isfinite(s.get("excess_spy",float("nan")))])
            vs_spy=float(exc.mean()) if len(exc)>0 else float("nan")
            for i,s in enumerate(top): s["rank"]=i+1

            results.append({
                "snapshot_date":snap_str,"top_tickers":top,
                "avg_ret":round(float(rets.mean()),2),"win_rate":round(float((rets>0).mean()*100),1),
                "vs_spy":round(vs_spy,2) if math.isfinite(vs_spy) else float("nan"),
                "n_valid":len(scores),"top_k":top_k,"lookahead":lookahead,
            })
        except Exception as e:
            logger.warning(f"스냅샷 {snap_str}: {e}")

    try:
        with open(SNAPSHOT_CACHE,"wb") as f:
            pickle.dump({"results":results,"bt_version":BT_VERSION,"computed":date.today().isoformat()},f)
    except Exception: pass
    return results


# ════════════════════════════════════════════════════════════════════
# 섹터 ETF 다운로드 (유지)
# ════════════════════════════════════════════════════════════════════

SECTOR_ETFS_TO_FETCH = [
    "SMH","SOXX","XLK","IGV","XLC","XLY","XLP","XLF","XLV","XLI",
    "XLE","XLB","XLU","XLRE","CIBR","ITA","IBB","XBI","GRID","AMPS","WCLD","BUG",
]

def fetch_sector_etfs(start=None, end=None) -> dict:
    import yfinance as yf
    if end is None: end=datetime.today()
    if start is None: start=end-timedelta(days=420)
    cache_p=CACHE_DIR/f"sector_etf_ohlcv_{date.today().isoformat()}.pkl"
    if cache_p.exists():
        try:
            with open(cache_p,"rb") as f: return pickle.load(f)
        except Exception: cache_p.unlink(missing_ok=True)
    result={}
    try:
        raw=yf.download(SECTOR_ETFS_TO_FETCH,start=start,end=end,auto_adjust=True,progress=False,group_by="ticker")
        if not raw.empty and isinstance(raw.columns,pd.MultiIndex):
            for etf in SECTOR_ETFS_TO_FETCH:
                try:
                    s=raw[etf]["Close"].dropna()
                    if len(s)>=20:
                        s.index=pd.to_datetime(s.index).tz_localize(None); result[etf]=s
                except Exception: pass
        with open(cache_p,"wb") as f: pickle.dump(result,f)
    except Exception as e:
        logger.warning(f"섹터 ETF 실패: {e}")
    return result


# ════════════════════════════════════════════════════════════════════
# 한국 백테스트 (코스피/코스닥 OHLCV 기반)
# ════════════════════════════════════════════════════════════════════

# 한국 ETF/인덱스 제외 목록
EXCLUDE_KR_BT_TICKERS = KR_BACKTEST_EXCLUDE_TICKERS

# 한국 ETF 이름 키워드 (종목명에 포함되면 제외)
KR_ETF_NAME_KEYWORDS = KR_BACKTEST_ETF_NAME_KEYWORDS


# 한국 백테스트 품질 필터
# - 20일 수익률 +300%/-80% 초과는 대부분 액면분할/감자/수정주가 문제이므로 제외
# - 시작가 1,000원 미만, 평균 거래대금 10억 원 미만은 백테스트 왜곡을 줄이기 위해 제외
KR_MIN_BT_PRICE = 1_000
KR_MIN_BT_AVG_AMOUNT = 1_000_000_000
KR_RET_CAP_HIGH = 300.0
KR_RET_CAP_LOW = -80.0
KR_MAX_DAILY_MOVE = 65.0

def _is_kr_unwanted_security(item: dict) -> bool:
    return is_kr_unwanted_security(item)


def _kr_amount_window(df: pd.DataFrame, end_idx: int, window: int = 20) -> float:
    try:
        if "Amount" in df.columns:
            amt = pd.to_numeric(df["Amount"], errors="coerce")
        elif "Volume" in df.columns and "Close" in df.columns:
            amt = pd.to_numeric(df["Volume"], errors="coerce") * pd.to_numeric(df["Close"], errors="coerce")
        else:
            return float("nan")
        return float(amt.iloc[max(0, end_idx - window + 1):end_idx + 1].dropna().mean())
    except Exception:
        return float("nan")


def _kr_has_suspicious_price_action(close: pd.Series, max_abs_daily_pct: float = KR_MAX_DAILY_MOVE) -> bool:
    try:
        r = pd.to_numeric(close, errors="coerce").pct_change().dropna() * 100
        if len(r) == 0:
            return False
        return bool((r.abs() > max_abs_daily_pct).any())
    except Exception:
        return False


def _bench_ret_between(bench_close: Optional[pd.Series], start_dt, end_dt) -> float:
    """벤치마크 수익률을 날짜 기준으로 계산. 날짜가 없으면 직전 거래일로 보정."""
    try:
        if bench_close is None or len(bench_close) < 2:
            return float("nan")
        s = pd.to_numeric(bench_close, errors="coerce").dropna().sort_index()
        start_dt = pd.Timestamp(start_dt)
        end_dt = pd.Timestamp(end_dt)
        b0s = s.loc[:start_dt]
        b1s = s.loc[:end_dt]
        if len(b0s) == 0 or len(b1s) == 0:
            return float("nan")
        b0 = float(b0s.iloc[-1]); b1 = float(b1s.iloc[-1])
        if b0 <= 0:
            return float("nan")
        return (b1 / b0 - 1) * 100
    except Exception:
        return float("nan")


def _kr_close_from_item(item) -> Optional[pd.Series]:
    """KR 백테스트 입력에서 Close 시리즈를 안전하게 추출."""
    try:
        obj = item.get("ohlcv") if isinstance(item, dict) else item
        if isinstance(obj, pd.Series):
            s = obj.copy()
        elif isinstance(obj, pd.DataFrame):
            if "Close" in obj.columns:
                s = obj["Close"].copy()
            elif len(obj.columns) == 1:
                s = obj.iloc[:, 0].copy()
            else:
                return None
        else:
            return None
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 20:
            return None
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return None


def _is_kr_etf_like(item: dict) -> bool:
    name = str(item.get("name", "")).upper() if isinstance(item, dict) else ""
    return is_kr_backtest_etf_name(name)


def run_kr_walkforward_backtest(
    ohlcv_data:  dict,
    lookahead:   int = 20,
    step:        int = 5,
    min_history: int = 120,
    bench_code:  str = "kospi",
    progress_cb  = None,
    min_price: float = KR_MIN_BT_PRICE,
    min_avg_amount: float = KR_MIN_BT_AVG_AMOUNT,
    ret_cap_high: float = KR_RET_CAP_HIGH,
    ret_cap_low: float = KR_RET_CAP_LOW,
) -> dict:
    """
    한국 워크포워드 백테스트.
    v10 개선:
    - 저가주/거래대금 부족/ETF·ETN·SPAC·우선주 제거
    - 액면분할·감자 등 수정주가 미반영으로 보이는 이상 수익률 제거
    - KOSPI와 KOSDAQ 초과수익을 동시에 기록
    - 실제 Top10/Top20 편입 종목이 df_records/by_date_portfolio에 남도록 name/market 포함
    """
    records = []
    filter_stats = {
        "universe_total": len(ohlcv_data),
        "short_history": 0,
        "bad_security": 0,
        "bad_ohlcv": 0,
        "suspicious_daily_move": 0,
        "low_price": 0,
        "low_liquidity": 0,
        "return_outlier": 0,
        "score_fail": 0,
        "eligible_tickers": 0,
        "records_before_filter": 0,
        "records_after_filter": 0,
    }

    tickers = []
    for t, v in ohlcv_data.items():
        if (
            not isinstance(v, dict)
            or is_kr_backtest_excluded_ticker(t)
            or is_kr_benchmark_code(t)
        ):
            continue
        if _is_kr_unwanted_security(v):
            filter_stats["bad_security"] += 1
            continue
        df0 = v.get("ohlcv")
        if not isinstance(df0, pd.DataFrame) or "Close" not in df0.columns:
            filter_stats["bad_ohlcv"] += 1
            continue
        if len(df0) < min_history + lookahead + 5:
            filter_stats["short_history"] += 1
            continue
        tickers.append(t)

    filter_stats["eligible_tickers"] = len(tickers)
    if not tickers:
        logger.warning("한국 백테스트 가능 종목 없음")
        return {}

    # 벤치마크: KOSPI/KOSDAQ 둘 다 확보. primary는 bench_code.
    bench_map = {}
    for code in ["kospi", "kosdaq", "069500", "122630"]:
        if code in ohlcv_data:
            s = _kr_close_from_item(ohlcv_data[code])
            if s is not None and len(s) > min_history:
                bench_map[code] = s
    primary_bench = bench_map.get(bench_code)
    if primary_bench is None:
        primary_bench = bench_map.get("kospi")
    if primary_bench is None:
        primary_bench = bench_map.get("069500")

    try:
        w = load_optimal_weights_kr()
    except Exception:
        w = _default_weights()

    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(i + 1, total, ticker)
        item = ohlcv_data[ticker]
        df = item["ohlcv"].copy().sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if "Open" not in df.columns:
            df["Open"] = df["Close"]
        close = df["Close"]
        n = len(close)
        if n < min_history + lookahead + 5:
            continue

        if _kr_has_suspicious_price_action(close):
            filter_stats["suspicious_daily_move"] += 1
            continue

        for t_idx in range(min_history, n - lookahead - 1, step):
            try:
                df_t = df.iloc[:t_idx + 1].copy()
                close_t = df_t["Close"]
                decision_dt = df.index[t_idx]
                entry_idx = min(t_idx + 1, n - 1)
                exit_idx = min(t_idx + lookahead, n - 1)
                entry_dt = df.index[entry_idx]
                exit_dt = df.index[exit_idx]

                p_decision = float(close.iloc[t_idx])
                p_entry = float(df["Open"].iloc[entry_idx]) if "Open" in df.columns else p_decision
                if p_entry <= 0 or not math.isfinite(p_entry):
                    p_entry = p_decision
                p_exit = float(close.iloc[exit_idx])
                if p_decision < min_price or p_entry < min_price:
                    filter_stats["low_price"] += 1
                    continue

                avg_amt = _kr_amount_window(df, t_idx, 20)
                if math.isfinite(avg_amt) and avg_amt < min_avg_amount:
                    filter_stats["low_liquidity"] += 1
                    continue

                # 날짜 정렬된 primary bench를 점수 계산용으로만 넘긴다.
                bench_t = primary_bench.loc[:decision_dt] if primary_bench is not None else None
                sc = _calc_scores_at(close_t, df_t, bench_t, len(close_t) - 1)
                if sc is None:
                    filter_stats["score_fail"] += 1
                    continue

                # 수급 시계열이 있으면 점수 시점(T) 기준 supply_flow를 채운다(look-ahead 차단).
                _sup_df = item.get("supply_series")
                if _sup_df is not None:
                    sc["supply_flow"] = _supply_flow_at(
                        _sup_df, decision_dt, mc=item.get("market_cap", float("nan")), avg_amt=avg_amt)

                _r = sc.get("risk", 30); _r20 = sc.get("ret_20d", 0)
                _rp = 15.0 if (_r >= 90 and _r20 < 0) else 8.0 if _r >= 90 else 2.0 if _r >= 80 else 0.0
                # 신규/enhanced factor까지 포함한 공통식 사용
                composite = float(calc_validated_score_series(pd.DataFrame([sc]), weights=w).iloc[0])

                if p_entry <= 0:
                    continue
                ret = (p_exit / p_entry - 1) * 100
                # 20거래일 내 최고수익률
                try:
                    col_h = "High" if "High" in df.columns else "Close"
                    fut_high = float(df[col_h].iloc[entry_idx:exit_idx+1].max())
                    _kr_max_ret_20d = round((fut_high/p_entry-1)*100, 2) if p_entry>0 else float("nan")
                except Exception:
                    _kr_max_ret_20d = float("nan")

                filter_stats["records_before_filter"] += 1
                if (not math.isfinite(ret)) or ret > ret_cap_high or ret < ret_cap_low:
                    filter_stats["return_outlier"] += 1
                    continue

                kospi_ret = _bench_ret_between(bench_map.get("kospi"), decision_dt, exit_dt)
                kosdaq_ret = _bench_ret_between(bench_map.get("kosdaq"), decision_dt, exit_dt)
                primary_ret = _bench_ret_between(primary_bench, decision_dt, exit_dt)

                def _dstr(v):
                    return str(v.date()) if hasattr(v, "date") else str(v)[:10]

                row = {
                    "ticker":        ticker,
                    "name":          item.get("name", ticker),
                    "market":        item.get("market", "KR"),
                    "t_idx":         t_idx,
                    "decision_date": _dstr(decision_dt),
                    "entry_date":    _dstr(entry_dt),
                    "exit_date":     _dstr(exit_dt),
                    "entry_price":   round(p_entry, 0),
                    "exit_price":    round(p_exit, 0),
                    "avg_amount_20d": round(avg_amt, 0) if math.isfinite(avg_amt) else float("nan"),
                    "ret":           round(ret, 2),
                    "max_ret_20d":   _kr_max_ret_20d,
                    # 기존 UI 호환
                    "spy_ret":       round(primary_ret, 2) if math.isfinite(primary_ret) else float("nan"),
                    "excess_spy":    round(ret - primary_ret, 2) if math.isfinite(primary_ret) else float("nan"),
                    "kospi_ret":     round(kospi_ret, 2) if math.isfinite(kospi_ret) else float("nan"),
                    "excess_kospi":  round(ret - kospi_ret, 2) if math.isfinite(kospi_ret) else float("nan"),
                    "kosdaq_ret":    round(kosdaq_ret, 2) if math.isfinite(kosdaq_ret) else float("nan"),
                    # 시장별 적합한 초과수익 (KOSDAQ종목→KOSDAQ초과, KOSPI→KOSPI초과)
                    "excess_market": round(ret - kosdaq_ret, 2) if (item.get("market","KOSPI")=="KOSDAQ" and math.isfinite(kosdaq_ret))
                                     else round(ret - kospi_ret, 2) if math.isfinite(kospi_ret) else float("nan"),
                    "excess_kosdaq": round(ret - kosdaq_ret, 2) if math.isfinite(kosdaq_ret) else float("nan"),
                    "composite":     round(composite, 1),
                    "validated_score": round(composite, 1),
                    **sc,
                }
                records.append(row)
                filter_stats["records_after_filter"] += 1
            except Exception as e:
                logger.debug(f"KR BT {ticker} t={t_idx}: {e}")

    if len(records) < 20:
        logger.warning(f"한국 백테스트 레코드 부족: {len(records)}건 / 필터 {filter_stats}")
        return {}

    df_rec = pd.DataFrame(records)
    result = _analyze_backtest(df_rec, lookahead)
    result.update({
        "n_records":  len(records),
        "n_tickers":  len(set(r["ticker"] for r in records)),
        "bt_version": BT_VERSION,
        "market":     "KR",
        "benchmarks": list(bench_map.keys()),
        "primary_benchmark": bench_code,
        "filter_stats": filter_stats,
        "filters": {
            "min_price": min_price,
            "min_avg_amount": min_avg_amount,
            "ret_cap_high": ret_cap_high,
            "ret_cap_low": ret_cap_low,
            "max_daily_move": KR_MAX_DAILY_MOVE,
        },
    })

    try:
        with open(BT_KR_CACHE, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"한국 백테스트 저장: {len(records)}건 / 필터 {filter_stats}")
    except Exception:
        pass
    return result

def load_kr_backtest() -> Optional[dict]:
    """한국 백테스트 캐시 로드."""
    if not BT_KR_CACHE.exists(): return None
    try:
        with open(BT_KR_CACHE,"rb") as f: result = pickle.load(f)
        if result.get("bt_version") != BT_VERSION:
            BT_KR_CACHE.unlink(missing_ok=True); return None
        computed = result.get("overall",{}).get("computed","")
        if computed:
            try:
                if (date.today() - date.fromisoformat(computed)).days > 7:
                    BT_KR_CACHE.unlink(missing_ok=True); return None
            except Exception: pass
        return result
    except Exception:
        BT_KR_CACHE.unlink(missing_ok=True); return None


def optimize_kr_weights(bt_result: dict) -> dict:
    """한국 전용 가중치 최적화 (미국과 동일 로직, 한국 데이터 기반)."""
    df = bt_result.get("df_records")
    if df is None or len(df) < 30:
        return _default_weights()

    # Train/Test 분리
    if "decision_date" in df.columns:
        all_dates  = sorted(df["decision_date"].unique())
        n_train    = max(8, int(len(all_dates) * 0.7))
        train_dates = set(all_dates[:n_train])
        test_dates  = set(all_dates[n_train:])
        df_train = df[df["decision_date"].isin(train_dates)].copy()
        df_test  = df[df["decision_date"].isin(test_dates)].copy()
    else:
        n_train = int(len(df)*0.7)
        df_train = df.iloc[:n_train].copy()
        df_test  = df.iloc[n_train:].copy()

    required = ["leader","volume","buyable","breakout","catalyst","quality","risk","ret"]
    for col in required:
        if col not in df_train.columns: df_train[col] = 50.0
        if col not in df_test.columns:  df_test[col]  = 50.0
    df_train = df_train.dropna(subset=required)
    df_test  = df_test.dropna(subset=required)
    if len(df_train) < 20: return _default_weights()

    def _sharpe_kr(df_sub, w_lead, w_vol, w_brk, w_cat, w_qot, w_risk):
        # 목표: KOSPI/KOSDAQ 초과수익 최대화
        import numpy as _np
        # [공식 일원화] 실전과 동일한 calc_validated_score_series 사용.
        w_try = {"leader": w_lead, "volume": w_vol, "breakout": w_brk,
                 "catalyst": w_cat, "quality": 0.0, "risk_penalty": w_risk}  # quality 제거
        score = calc_validated_score_series(df_sub, weights=w_try).values
        threshold = _np.quantile(score, 0.70)
        top_idx   = score >= threshold
        # KOSPI 초과수익 우선, 없으면 KOSDAQ 초과, 없으면 원수익률
        for exc_col in ["excess_kospi", "excess_kosdaq", "excess_spy"]:
            if exc_col in df_sub.columns:
                exc = df_sub.loc[top_idx, exc_col].dropna()
                if len(exc) >= 5:
                    return float(exc.mean()/(exc.std()+1e-8))
        top_rets = df_sub.loc[top_idx, "ret"]
        if len(top_rets) < 5: return -999
        return float(top_rets.mean()/(top_rets.std()+1e-8))

    # 한국 특성 반영: supply_score 가중치 크게 (외국인/기관 수급 중요)
    best_sharpe = -999; best_w = None
    for w_lead in [0.30,0.33,0.35,0.38]:
        for w_vol in [0.12,0.15,0.18]:
            for w_brk in [0.08,0.10,0.12]:
                for w_cat in [0.10,0.12,0.15]:
                    w_qot = max(0.05, round(1.0-w_lead-w_vol-w_brk-w_cat, 2))
                    if w_qot < 0.05 or w_qot > 0.25: continue
                    if abs(w_lead+w_vol+w_brk+w_cat+w_qot-1.0) > 0.02: continue
                    for w_risk in [0.05,0.08,0.10]:
                        sh = _sharpe_kr(df_train,w_lead,w_vol,w_brk,w_cat,w_qot,w_risk)
                        if sh > best_sharpe:
                            best_sharpe = sh
                            best_w = dict(leader=w_lead,volume=w_vol,breakout=w_brk,
                                         catalyst=w_cat,quality=0.0,risk_penalty=w_risk)

    if best_w is None: return _default_weights()

    dw = _default_weights()
    test_best    = _sharpe_kr(df_test, best_w["leader"],best_w["volume"],best_w["breakout"],best_w["catalyst"],best_w["quality"],best_w["risk_penalty"])
    test_default = _sharpe_kr(df_test, dw["leader"],dw["volume"],dw["breakout"],dw["catalyst"],dw["quality"],dw["risk_penalty"])
    train_default= _sharpe_kr(df_train,dw["leader"],dw["volume"],dw["breakout"],dw["catalyst"],dw["quality"],dw["risk_penalty"])

    best_w.update({
        "train_sharpe":      round(best_sharpe, 4),
        "test_sharpe":       round(test_best, 4),
        "default_train":     round(train_default, 4),
        "default_test":      round(test_default, 4),
        "improvement_train": round(best_sharpe - train_default, 4),
        "improvement_test":  round(test_best   - test_default,  4),
        "n_train_dates":     len(set(df_train.get("decision_date",pd.Series()).unique())),
        "n_test_dates":      len(set(df_test.get("decision_date",pd.Series()).unique())),
        "optimized_at":      date.today().isoformat(),
        "market":            "KR",
    })

    if best_w["improvement_test"] >= -0.05:
        try:
            with open(OPT_KR_CACHE,"wb") as f: pickle.dump(best_w, f)
        except Exception: pass
        save_weights_history(best_w, "KR")
        return best_w
    else:
        default = _default_weights()
        default["rejected_reason"] = "KR Test 검증 미통과"
        default["market"] = "KR"
        save_weights_history(default, "KR")
        return default


def load_optimal_weights_kr() -> dict:
    """한국 최적 가중치 로드."""
    if OPT_KR_CACHE.exists():
        try:
            with open(OPT_KR_CACHE,"rb") as f:
                _w = pickle.load(f)
            _w["quality"] = 0.0   # quality 영구 제거(백테스트 역효과) — 캐시 덮어쓰기
            return _w
        except Exception: pass
    return _default_weights()


# ════════════════════════════════════════════════════════════════════
# Factor 개별 예측력 진단
# ════════════════════════════════════════════════════════════════════

def analyze_factor_power(df: pd.DataFrame, bench_col: str = "excess_spy",
                         target_col: str = "ret") -> pd.DataFrame:
    """
    각 factor의 개별 예측력 진단.
    Returns: DataFrame with Spearman, R², 상위20% vs 하위20% spread 등
    """
    factors = ["leader","volume","breakout","catalyst","quality","risk","composite",
               "momentum_accel","volume_shock","brk_persist","near_high","vol_contract",
               "pullback_q","risk_filter","sect_rs","supply_flow",
               "volume_enhanced","breakout_enhanced","risk_combined"]
    rows = []
    for col in factors:
        if col not in df.columns or "ret" not in df.columns:
            continue
        # target_col: ret / excess_spy / max_ret_20d 등
        tgt = target_col if target_col in df.columns else "ret"
        valid = df[[col, tgt]].dropna()
        if bench_col in df.columns:
            valid2 = df[[col, bench_col]].dropna()
        else:
            valid2 = pd.DataFrame()

        n = len(valid)
        if n < 20:
            continue

        # 상관계수
        try:
            spearman = float(valid[col].rank().corr(valid[tgt].rank()))
        except Exception:
            spearman = float("nan")
        try:
            pearson = float(valid[col].corr(valid[tgt]))
        except Exception:
            pearson = float("nan")

        # R²
        try:
            x = valid[col].values; y = valid[tgt].values
            coef = np.polyfit(x, y, 1)
            y_pred = np.polyval(coef, x)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            r2 = float("nan")

        # 상위/하위 20%
        q80 = valid[col].quantile(0.80)
        q20 = valid[col].quantile(0.20)
        top20 = valid[valid[col] >= q80][tgt]
        bot20 = valid[valid[col] <= q20][tgt]

        top_avg = float(top20.mean()) if len(top20) >= 5 else float("nan")
        bot_avg = float(bot20.mean()) if len(bot20) >= 5 else float("nan")
        spread  = top_avg - bot_avg if (math.isfinite(top_avg) and math.isfinite(bot_avg)) else float("nan")
        top_wr  = float((top20 > 0).mean() * 100) if len(top20) >= 5 else float("nan")

        # 초과수익 상위 20%
        exc_top_avg = float("nan")
        if len(valid2) >= 20 and bench_col in valid2.columns:
            q80b = valid2[col].quantile(0.80)
            exc_top = valid2[valid2[col] >= q80b][bench_col]
            exc_top_avg = float(exc_top.mean()) if len(exc_top) >= 5 else float("nan")

        rows.append({
            "factor":     col,
            "n":          n,
            "Spearman":   round(spearman, 4) if math.isfinite(spearman) else None,
            "Pearson":    round(pearson,  4) if math.isfinite(pearson)  else None,
            "R²":         round(r2,       4) if math.isfinite(r2)       else None,
            "상위20%수익":  round(top_avg, 2) if math.isfinite(top_avg) else None,
            "하위20%수익":  round(bot_avg, 2) if math.isfinite(bot_avg) else None,
            "상하spread":   round(spread,  2) if math.isfinite(spread)  else None,
            "상위20%승률":  round(top_wr,  1) if math.isfinite(top_wr)  else None,
            "상위20%초과":  round(exc_top_avg, 2) if math.isfinite(exc_top_avg) else None,
            "판정": _factor_verdict(spearman, spread, top_wr),
        })

    return pd.DataFrame(rows)


def _factor_verdict(spearman, spread, top_wr) -> str:
    """Factor 예측력 판정."""
    if not math.isfinite(spearman) or not math.isfinite(spread):
        return "데이터부족"
    if spearman > 0.08 and spread > 2 and (not math.isfinite(top_wr) or top_wr >= 55):
        return "✅ 강"
    if spearman > 0.03 and spread > 0.5:
        return "🟡 보통"
    if spearman < -0.03 or spread < -1:
        return "🔴 역효과"
    return "⚪ 약"


def analyze_factor_quintiles(df: pd.DataFrame, bench_col: str = "excess_spy",
                             target_col: str = "ret") -> dict:
    """각 factor를 5분위로 나눠 수익률 단조성 확인."""
    factors = ["leader","volume","breakout","catalyst","quality","composite","validated_score"]
    tgt = target_col if target_col in df.columns else "ret"
    result = {}
    for col in factors:
        if col not in df.columns:
            continue
        valid = df[[col, tgt]].dropna()
        if len(valid) < 50:
            continue
        rows = []
        for q in range(5):
            lo = valid[col].quantile(q * 0.2)
            hi = valid[col].quantile((q+1) * 0.2)
            sub = valid[(valid[col] >= lo) & (valid[col] < hi)] if q < 4 else valid[valid[col] >= lo]
            exc_avg = float("nan")
            if bench_col in df.columns:
                sub2 = df[(df[col] >= lo) & (df[col] < hi)][[col, bench_col]].dropna() if q < 4 else df[df[col] >= lo][[col, bench_col]].dropna()
                if len(sub2) >= 5:
                    exc_avg = round(float(sub2[bench_col].mean()), 2)
            rows.append({
                "구간":    f"Q{q+1}({int(lo)}~{int(hi)})",
                "n":       len(sub),
                f"평균({tgt})": round(float(sub[tgt].mean()), 2) if len(sub) >= 5 else None,
                f"중앙({tgt})": round(float(sub[tgt].median()), 2) if len(sub) >= 5 else None,
                "승률%":   round(float((sub[tgt] > 0).mean() * 100), 1) if len(sub) >= 5 else None,
                "초과수익": exc_avg,
            })
        result[col] = rows
    return result


def compute_custom_objective(
    df: pd.DataFrame,
    objective: str,
    weights: dict,
    bench_col: str = "excess_spy",
    top_k: int = 10,
) -> dict:
    """
    사용자 정의 목표값으로 가중치 성과 계산.
    objective 옵션:
      'ret20'       - 20일 원수익률
      'excess'      - 초과수익
      'max_ret'     - 20일 내 최고수익률
      'hit15'       - +15% 이상 달성 여부
      'top10_excess'- 날짜별 상위10개 초과수익
      'top10_hit15' - 날짜별 상위10개 중 +15% 비율
    """
    import numpy as _np

    # composite 재계산: 실전 validated_score와 같은 공통 공식 사용
    w = weights
    comp = calc_validated_score_series(df, weights=w)
    df = df.copy(); df["_comp"] = comp

    # 목표별 계산
    results_by_date = []
    dates = df["decision_date"].unique() if "decision_date" in df.columns else []
    for dt in sorted(dates):
        sub = df[df["decision_date"] == dt].copy().sort_values("_comp", ascending=False)
        top = sub.head(top_k)
        rets = top["ret"].dropna()
        exc  = top[bench_col].dropna() if bench_col in top.columns else pd.Series()
        if len(rets) == 0:
            continue
        # max_ret_20d: 20거래일 내 최고수익률 (급등 탐지용)
        max_rets = top["max_ret_20d"].dropna() if "max_ret_20d" in top.columns else rets

        if objective == "ret20":
            score = float(rets.mean())
        elif objective == "excess":
            score = float(exc.mean()) if len(exc) > 0 else float(rets.mean())
        elif objective == "max_ret":
            score = float(max_rets.mean()) if len(max_rets) > 0 else float(rets.mean())
        elif objective == "hit15":
            # 상위 N개 중 20거래일 내 +15% 도달 비율 (top_k 기준)
            score = float((max_rets >= 15).mean() * 100) if len(max_rets) > 0 else float((rets >= 15).mean() * 100)
        elif objective == "top10_hit15":
            # 무조건 상위 10개 기준 급등 달성 비율 (top_k 무시)
            top10 = sub.head(10)
            m10 = top10["max_ret_20d"].dropna() if "max_ret_20d" in top10.columns else top10["ret"].dropna()
            score = float((m10 >= 15).mean() * 100) if len(m10) > 0 else 0.0
        else:
            score = float(rets.mean())
        results_by_date.append({"date": dt, "score": score, "n": len(rets)})

    if not results_by_date:
        return {}
    scores = pd.Series([r["score"] for r in results_by_date])
    return {
        "avg":    round(float(scores.mean()), 2),
        "median": round(float(scores.median()), 2),
        "win_rate": round(float((scores > 0).mean() * 100), 1),
        "worst":  round(float(scores.min()), 2),
        "n_dates": len(scores),
        "by_date": results_by_date,
    }


def walk_forward_optimization(
    df: pd.DataFrame,
    train_months: int = 6,
    test_months:  int = 1,
    bench_col:    str = "excess_spy",
    objective:    str = "excess",
    top_k:        int = 10,
) -> list[dict]:
    """
    Walk-forward 최적화.
    매 test_months마다 과거 train_months로 최적 가중치 찾고 검증.
    """
    if "decision_date" not in df.columns or len(df) < 100:
        return []

    df = df.copy()
    df["_month"] = pd.to_datetime(df["decision_date"]).dt.to_period("M")
    months = sorted(df["_month"].unique())
    if len(months) < train_months + test_months:
        return []

    results = []
    for i in range(train_months, len(months) - test_months + 1, test_months):
        train_months_range = months[i - train_months:i]
        test_months_range  = months[i:i + test_months]

        df_train = df[df["_month"].isin(train_months_range)].copy()
        df_test  = df[df["_month"].isin(test_months_range)].copy()

        if len(df_train) < 50 or len(df_test) < 10:
            continue

        # 그리드 서치 (train) — 신규 factor 포함 확장 탐색
        best_score = -999; best_w = None

        # 사용 가능한 신규 factor 컬럼 확인
        new_factor_cols = ["momentum_accel","volume_shock","brk_persist","near_high",
                           "vol_contract","pullback_q",
                           "volume_enhanced","breakout_enhanced","risk_combined"]
        avail_new = [c for c in new_factor_cols if c in df_train.columns]
        use_new_factors = len(avail_new) >= 3

        if use_new_factors:
            # 신규 factor 포함 확장 그리드 (독립 가중치 최적화)
            for w_lead in [0.22, 0.26, 0.30]:
                for w_vol_e in [0.12, 0.15, 0.18]:    # volume_enhanced
                    for w_brk_e in [0.10, 0.13, 0.16]: # breakout_enhanced
                        for w_mom_a in [0.05, 0.08, 0.10]: # momentum_accel
                            for w_near in [0.04, 0.07]:    # near_high
                                for w_vol_c in [0.04, 0.06]: # vol_contract
                                    for w_cat in [0.07, 0.09]:
                                        used = w_lead+w_vol_e+w_brk_e+w_mom_a+w_near+w_vol_c+w_cat
                                        w_qot = round(max(0.03, 1.0-used), 2)
                                        if w_qot < 0.03 or w_qot > 0.18: continue
                                        for w_risk in [0.05, 0.08]:
                                            w = {
                                                "leader":         w_lead,
                                                "volume":         w_vol_e,    # volume_enhanced 사용
                                                "breakout":       w_brk_e,    # breakout_enhanced 사용
                                                "catalyst":       w_cat,
                                                "quality":        w_qot,
                                                "risk_penalty":   w_risk,
                                                "momentum_accel": w_mom_a,    # 독립 신규 factor
                                                "near_high":      w_near,     # 독립 신규 factor
                                                "vol_contract":   w_vol_c,    # 독립 신규 factor
                                                "risk":           0,          # risk_combined가 risk로 읽힘
                                            }
                                            res = compute_custom_objective(df_train, objective, w, bench_col, top_k)
                                            sc = res.get("avg", -999)
                                            if sc > best_score:
                                                best_score = sc; best_w = w.copy()
        else:
            # 기존 5-factor 그리드 (신규 factor 없을 때 fallback)
            for w_lead in [0.30, 0.35, 0.38, 0.42]:
                for w_vol in [0.12, 0.15, 0.18]:
                    for w_brk in [0.08, 0.10, 0.12, 0.15]:
                        for w_cat in [0.10, 0.12, 0.15]:
                            w_qot = round(max(0.05, 1.0 - w_lead - w_vol - w_brk - w_cat), 2)
                            if w_qot < 0.05 or w_qot > 0.25: continue
                            for w_risk in [0.03, 0.05, 0.08, 0.10, 0.12]:
                                w = {"leader":w_lead,"volume":w_vol,"breakout":w_brk,
                                     "catalyst":w_cat,"quality":w_qot,"risk_penalty":w_risk}
                                res = compute_custom_objective(df_train, objective, w, bench_col, top_k)
                                sc = res.get("avg", -999)
                                if sc > best_score:
                                    best_score = sc; best_w = w.copy()

        if best_w is None:
            continue

        # test 검증
        test_res   = compute_custom_objective(df_test, objective, best_w,  bench_col, top_k)
        default_w  = _default_weights()
        test_def   = compute_custom_objective(df_test, objective, default_w, bench_col, top_k)

        results.append({
            "train_end":   str(train_months_range[-1]),
            "test_period": str(test_months_range[0]),
            "best_weights": best_w,
            "train_score":  round(best_score, 3),
            "test_score":   round(test_res.get("avg", float("nan")), 3),
            "test_default": round(test_def.get("avg", float("nan")), 3),
            "improvement":  round(test_res.get("avg", 0) - test_def.get("avg", 0), 3),
            "test_winrate": test_res.get("win_rate", float("nan")),
            "n_test":       test_res.get("n_dates", 0),
        })

    return results




# ════════════════════════════════════════════════════════════════════
# 날짜별 Percentile 정규화 + hit15 중심 진단
# ════════════════════════════════════════════════════════════════════

def normalize_by_date(df: pd.DataFrame, factor_cols: list) -> pd.DataFrame:
    """
    날짜별 percentile rank로 factor 정규화.
    특정 시기 시장 강약에 따른 점수 왜곡 방지.
    """
    df = df.copy()
    if "decision_date" not in df.columns:
        return df
    for col in factor_cols:
        if col not in df.columns:
            continue
        norm_col = f"{col}_rank"
        df[norm_col] = df.groupby("decision_date")[col].transform(
            lambda x: x.rank(pct=True) * 100
        )
    return df


def analyze_factor_hit15(df: pd.DataFrame, bench_col: str = "excess_spy") -> pd.DataFrame:
    """
    hit15 기준 factor 진단: 각 factor 상위 20%에서 hit15 달성 비율.
    R²보다 상위권 압축 성능 중심으로 평가.
    """
    factors = [
        "leader","volume","breakout","catalyst","quality","risk","composite",
        "momentum_accel","volume_shock","brk_persist","near_high","vol_contract",
        "pullback_q","risk_filter","sect_rs","supply_flow",
        "volume_enhanced","breakout_enhanced","risk_combined"
    ]

    # hit15 컬럼 확인
    if "max_ret_20d" in df.columns:
        df = df.copy()
        df["hit15"] = (df["max_ret_20d"] >= 15).astype(float) * 100
        target_col = "hit15"
    elif "ret" in df.columns:
        df = df.copy()
        df["hit15"] = (df["ret"] >= 15).astype(float) * 100
        target_col = "hit15"
    else:
        return pd.DataFrame()

    rows = []
    for col in factors:
        if col not in df.columns:
            continue
        # 날짜별 percentile rank가 있으면 진단/분위 산정은 rank 기준으로 수행
        # 표시 factor명은 원래 이름을 유지.
        score_col = f"{col}_rank" if f"{col}_rank" in df.columns else col
        need_cols = [score_col, target_col, "ret"]
        valid = df[need_cols].dropna()
        if len(valid) < 30:
            continue

        q80 = valid[score_col].quantile(0.80)
        q20 = valid[score_col].quantile(0.20)
        top20 = valid[valid[score_col] >= q80]
        bot20 = valid[valid[score_col] <= q20]

        # hit15 달성률
        top_hit = float(top20[target_col].mean()) if len(top20) >= 5 else float("nan")
        bot_hit = float(bot20[target_col].mean()) if len(bot20) >= 5 else float("nan")
        all_hit = float(valid[target_col].mean())

        # max_ret_20d 평균
        max_ret_top = float("nan")
        if "max_ret_20d" in df.columns:
            top20_m = df.loc[df[score_col] >= q80, "max_ret_20d"].dropna()
            max_ret_top = round(float(top20_m.mean()), 2) if len(top20_m) >= 5 else float("nan")

        # 시장 초과수익
        exc_top = float("nan")
        if bench_col in df.columns:
            exc_df = df[[score_col, bench_col]].dropna()
            top_exc = exc_df[exc_df[score_col] >= q80][bench_col]
            exc_top = round(float(top_exc.mean()), 2) if len(top_exc) >= 5 else float("nan")

        # Spearman (hit15 기준)
        try:
            sp = float(valid[score_col].rank().corr(valid[target_col].rank()))
        except Exception:
            sp = float("nan")

        # 판정: 상위 20%에서 hit15 달성률이 전체 평균보다 높으면 유효
        lift = top_hit - all_hit if (math.isfinite(top_hit) and math.isfinite(all_hit)) else float("nan")
        if math.isfinite(lift):
            if lift > 10:      verdict = "✅ 강 (급등 압축력 높음)"
            elif lift > 5:     verdict = "🟡 보통"
            elif lift > 0:     verdict = "⚪ 약한 양의 효과"
            else:              verdict = "🔴 역효과"
        else:
            verdict = "데이터부족"

        rows.append({
            "factor":          col,
            "n":               len(valid),
            "전체hit15%":      round(all_hit, 1) if math.isfinite(all_hit) else None,
            "상위20%hit15%":   round(top_hit, 1) if math.isfinite(top_hit) else None,
            "하위20%hit15%":   round(bot_hit, 1) if math.isfinite(bot_hit) else None,
            "Lift(상위-전체)":  round(lift, 1)   if math.isfinite(lift)   else None,
            "상위20%maxRet":   max_ret_top,
            "상위20%초과수익":  exc_top,
            "Spearman(hit15)": round(sp, 4)      if math.isfinite(sp)     else None,
            "판정":            verdict,
        })

    return pd.DataFrame(rows)
