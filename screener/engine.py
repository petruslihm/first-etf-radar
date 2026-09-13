"""
screener/engine.py  v7.1

추가:
  1. 섹터 ETF 수익률 기반 강한 섹터 자동 분석
  2. 재무를 파이프라인에 통합 (1차 150개 → 재무 → 최종 Top)
  3. 테마 강도를 전체 유니버스 기반으로 계산
  4. Turnaround 트랙 (52주 저점 반등형) 추가
"""

import logging
import math
import pickle
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from screener.factors import (
    calc_kr_factors, calc_us_factors,
    market_regime_score, _ret,
)
from screener.universe import (
    KR_FILTER, SECTOR_ETF, US_FILTER, normalize_sector,
)
from screener.filters import (
    is_kr_pipeline_etf_name,
    is_kr_preferred_ticker,
    is_us_live_etf_ticker,
)

logger    = logging.getLogger(__name__)
FACTOR_CACHE_VERSION = "v2_validated_pool"
from screener.paths import CACHE_DIR, TODAY, data_date_key, clear_data_date

# ── 트랙 분류 ─────────────────────────────────────────────────────
LEADER_TRACKS     = {"Hot Leader / Extended", "Hot Leader / Buyable", "Leader / Buyable"}
BUYABLE_TRACKS    = {"Leader / Buyable", "Leader / Pullback Wait", "Hot Leader / Buyable"}
BREAKOUT_TRACKS   = {"Breakout Signal"}
TURNAROUND_TRACKS = {"Turnaround / Early"}
AVOID_TRACKS      = {"Avoid / Weak"}

# ── 섹터 ETF 맵 (수익률 추적용) ───────────────────────────────────
SECTOR_ETF_MAP = {
    "반도체":     ["SMH", "SOXX"],
    "기술":       ["XLK", "IGV"],
    "AI/소프트웨어": ["IGV", "WCLD"],
    "통신":       ["XLC"],
    "소비재":     ["XLY", "XLP"],
    "금융":       ["XLF"],
    "헬스케어":   ["XLV"],
    "산업재":     ["XLI"],
    "에너지":     ["XLE"],
    "소재":       ["XLB"],
    "유틸리티":   ["XLU"],
    "부동산":     ["XLRE"],
    "사이버보안": ["CIBR", "BUG"],
    "전력인프라": ["GRID", "AMPS"],
    "방산":       ["ITA", "XAR"],
    "바이오":     ["IBB", "XBI"],
}


# ════════════════════════════════════════════════════════════════════
# 캐시
# ════════════════════════════════════════════════════════════════════

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"factors_{name}_{data_date_key()}_{FACTOR_CACHE_VERSION}.pkl"

def _load_cache(name: str):
    p = _cache_path(name)
    if p.exists():
        try:
            with open(p,"rb") as f: obj = pickle.load(f)
            logger.info(f"팩터 캐시 로드: {name} ({len(obj)}행)")
            return obj
        except Exception as e:
            logger.warning(f"캐시 손상({name}): {e}"); p.unlink(missing_ok=True)
    return None

def _save_cache(obj, name: str):
    try:
        with open(_cache_path(name),"wb") as f: pickle.dump(obj, f)
    except Exception as e:
        logger.warning(f"캐시 저장 실패({name}): {e}")

def clear_all_cache():
    # 날짜 키 제거에 맞춰 glob 패턴도 와일드카드로 (기준일 무관 전부 삭제)
    patterns = ["factors_us_*.pkl", "factors_kr_*.pkl",
                "us_tickers_*.pkl", "sector_etf_*.pkl",
                "dynamic_universe_*.pkl", "kr_ohlcv_*.pkl",
                "fund_us_*.pkl", "fund_kr_*.pkl"]
    cleared = 0
    for pat in patterns:
        for f in CACHE_DIR.glob(pat):
            f.unlink(missing_ok=True); cleared += 1
    clear_data_date()   # OHLCV를 지웠으므로 기준일도 리셋 → 다음 수집 때 새로 기록
    logger.info(f"캐시 {cleared}개 삭제")
    return cleared


def clear_scores_cache():
    """
    점수/팩터 캐시만 삭제하고 비싼 OHLCV 수집 캐시(kr_ohlcv_*)와 기준일은 보존한다.
    → 가중치·팩터 로직만 바꿨을 때 재수집 없이 점수만 빠르게 재계산하기 위함.
    (한국은 종목당 수십 페이지를 순차 스크래핑하므로 재수집이 매우 느림)
    """
    patterns = ["factors_us_*.pkl", "factors_kr_*.pkl", "sector_etf_*.pkl"]
    cleared = 0
    for pat in patterns:
        for f in CACHE_DIR.glob(pat):
            f.unlink(missing_ok=True); cleared += 1
    logger.info(f"점수 캐시 {cleared}개 삭제 (OHLCV 수집 캐시·기준일 보존)")
    return cleared


# ════════════════════════════════════════════════════════════════════
# 1. 섹터 ETF 수익률 분석
# ════════════════════════════════════════════════════════════════════

def analyze_sector_etfs(bench_close: dict) -> list[dict]:
    """
    섹터 ETF 수익률로 지금 강한/약한 섹터를 자동 분석.
    Returns: [{"sector": str, "etf": str, "ret5": float, "ret20": float,
               "ret60": float, "above_ma20": bool, "hot": bool}]
    """
    cache_p = CACHE_DIR / f"sector_etf_{data_date_key()}.pkl"
    if cache_p.exists():
        try:
            with open(cache_p,"rb") as f: return pickle.load(f)
        except Exception: cache_p.unlink(missing_ok=True)

    results = []
    for sector, etfs in SECTOR_ETF_MAP.items():
        best = None
        for etf in etfs:
            if etf in bench_close:
                best = (etf, bench_close[etf])
                break
        if best is None:
            continue
        etf, s = best
        try:
            r5  = float((s.iloc[-1]/s.iloc[-6]-1)*100)  if len(s)>5  else float("nan")
            r20 = float((s.iloc[-1]/s.iloc[-21]-1)*100) if len(s)>20 else float("nan")
            r60 = float((s.iloc[-1]/s.iloc[-61]-1)*100) if len(s)>60 else float("nan")
            ma20 = s.tail(20).mean()
            above_ma20 = bool(s.iloc[-1] > ma20) if math.isfinite(ma20) else False
            hot = (math.isfinite(r20) and r20 >= 5 and above_ma20)
            results.append({
                "sector": sector, "etf": etf,
                "ret5": round(r5,2) if math.isfinite(r5) else float("nan"),
                "ret20": round(r20,2) if math.isfinite(r20) else float("nan"),
                "ret60": round(r60,2) if math.isfinite(r60) else float("nan"),
                "above_ma20": above_ma20, "hot": hot,
            })
        except Exception:
            pass

    results.sort(key=lambda x: x.get("ret20", -999), reverse=True)
    try:
        with open(cache_p,"wb") as f: pickle.dump(results, f)
    except Exception: pass
    return results


# ════════════════════════════════════════════════════════════════════
# 2. 전체 유니버스 기반 테마 강도
# ════════════════════════════════════════════════════════════════════

def calc_universe_theme_strength(full_df: pd.DataFrame, market: str = "US") -> list[dict]:
    """
    전체 수집 종목(Top 결과 아닌 전체)의 수익률로 테마 강도 계산.
    """
    from screener.themes import US_THEME_MAP, KR_THEME_MAP
    theme_map = US_THEME_MAP if market == "US" else KR_THEME_MAP

    results = []
    for theme, members in theme_map.items():
        in_universe = [t for t in members if t in full_df.index]
        if len(in_universe) < 2:
            continue
        sub = full_df.loc[in_universe]
        avg_ret5  = float(sub["ret_5d"].dropna().mean())  if "ret_5d"  in sub.columns else 0
        avg_ret20 = float(sub["ret_20d"].dropna().mean()) if "ret_20d" in sub.columns else 0
        avg_mom   = float(sub["momentum_score"].dropna().mean()) if "momentum_score" in sub.columns else 50
        avg_lead  = float(sub["leader_score"].dropna().mean())   if "leader_score"   in sub.columns else 50

        # 상위 3종목
        if "ret_20d" in sub.columns:
            top3 = sub["ret_20d"].dropna().nlargest(3)
            name_col = "name" if "name" in sub.columns else "company"
            top_stocks = [(t,
                           str(sub.loc[t,name_col]) if name_col in sub.columns else t,
                           float(sub.loc[t,"ret_20d"])) for t in top3.index]
        else:
            top_stocks = []

        results.append({
            "theme":      theme,
            "count":      len(in_universe),
            "avg_ret5":   round(avg_ret5, 2),
            "avg_ret20":  round(avg_ret20, 2),
            "avg_mom":    round(avg_mom, 1),
            "avg_lead":   round(avg_lead, 1),
            "top_stocks": top_stocks,
            "hot":        avg_ret20 >= 5 and avg_mom >= 55,
        })

    results.sort(key=lambda x: x["avg_ret20"], reverse=True)
    return results


# ════════════════════════════════════════════════════════════════════
# 3. 투자경고 / 업종 매핑
# ════════════════════════════════════════════════════════════════════

def _fetch_warned_stocks() -> set:
    warned = set()
    try:
        import requests
        from bs4 import BeautifulSoup
        s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0"
        for url in ["https://finance.naver.com/sise/management.naver",
                    "https://finance.naver.com/sise/caution.naver"]:
            try:
                r = s.get(url, timeout=8); soup = BeautifulSoup(r.text,"html.parser")
                for a in soup.select("a[href*='code=']"):
                    code = a["href"].split("code=")[1][:6]
                    if code.isdigit() and len(code)==6: warned.add(code)
            except Exception: pass
    except Exception as e:
        logger.warning(f"투자경고 수집 실패: {e}")
    return warned

KR_SECTOR_FALLBACK = {
    "005":"전기전자","000":"금융","006":"화학","009":"철강금속",
    "010":"조선기계","012":"조선기계","015":"전기전자","017":"에너지화학",
    "020":"금융","028":"전기전자","030":"전기전자","033":"제약바이오",
    "034":"전기전자","035":"인터넷소프트","036":"인터넷소프트",
    "051":"제약바이오","066":"2차전지","068":"제약바이오",
    "086":"2차전지","096":"에너지화학","196":"제약바이오",
    "207":"제약바이오","247":"2차전지","267":"전력기기",
    "326":"제약바이오","352":"2차전지","361":"반도체","373":"2차전지",
}
def _get_kr_sector(code,raw): return raw.strip() if raw and raw.strip() else KR_SECTOR_FALLBACK.get(code[:3],"기타")

def _fetch_kr_sector_map() -> dict:
    sector_map = {}
    try:
        import requests, time
        from bs4 import BeautifulSoup
        s = requests.Session()
        s.headers.update({"User-Agent":"Mozilla/5.0","Referer":"https://finance.naver.com/"})
        r = s.get("https://finance.naver.com/sise/sise_group.naver",timeout=10)
        soup = BeautifulSoup(r.text,"html.parser")
        links = [(a["href"].split("no=")[1].split("&")[0],a.text.strip())
                 for a in soup.select("a[href*='sise_group_detail']")
                 if "no=" in a.get("href","") and a.text.strip()]
        for no,sname in links[:50]:
            try:
                r2 = s.get(f"https://finance.naver.com/sise/sise_group_detail.naver?type=T&no={no}",timeout=8)
                soup2 = BeautifulSoup(r2.text,"html.parser")
                for a in soup2.select("a[href*='code=']"):
                    code = a["href"].split("code=")[1][:6]
                    if code.isdigit() and len(code)==6: sector_map[code]=sname
                time.sleep(0.08)
            except Exception: pass
    except Exception as e:
        logger.warning(f"업종 매핑 실패: {e}")
    return sector_map


# ════════════════════════════════════════════════════════════════════
# 유틸
# ════════════════════════════════════════════════════════════════════

def _pct_rank(values: list) -> list:
    arr=np.array(values,dtype=float); valid=arr[np.isfinite(arr)]
    return [float((valid<v).sum())/max(len(valid)-1,1) if math.isfinite(v) else 0.5 for v in arr]

def _apply_theme_boost(df, theme_tickers, saved_themes):
    """
    테마 부스트 적용.
    saved_themes의 tickers가 str 또는 {"ticker":..,"role":..,"weight":..} 둘 다 처리.
    """
    if "is_theme_pick" not in df.columns: df["is_theme_pick"] = False
    if "theme_name"    not in df.columns: df["theme_name"]    = ""

    def _code(t):
        return t.get("ticker","") if isinstance(t, dict) else str(t)

    def _weight(t):
        return float(t.get("weight", 1.0)) if isinstance(t, dict) else 1.0

    # ticker → (theme명, weight) 매핑
    ticker_to_theme  = {}
    ticker_to_weight = {}
    for th in (saved_themes or []):
        for t in th.get("tickers", []):
            code = _code(t)
            if code:
                ticker_to_theme[code]  = th.get("name", "AI 추천")
                ticker_to_weight[code] = _weight(t)

    for ticker in (theme_tickers or []):
        if ticker not in df.index: continue
        w     = ticker_to_weight.get(ticker, 1.0)
        boost = round(10 * w, 1)   # leader=10, direct=~8, related=~5
        df.loc[ticker, "is_theme_pick"] = True
        df.loc[ticker, "theme_name"]    = ticker_to_theme.get(ticker, "AI 추천")
        base_score = float(df.loc[ticker, "final_score"] if "final_score" in df.columns else df.loc[ticker, "leader_final"])
        df.loc[ticker, "final_score"]   = min(100, base_score + boost)

    # 테마 내 20일 수익률 1등에 추가 +5
    for th in (saved_themes or []):
        members = [_code(t) for t in th.get("tickers", []) if _code(t) in df.index]
        if len(members) >= 2:
            best = df.loc[members, "ret_20d"].idxmax()
            base_score = float(df.loc[best, "final_score"] if "final_score" in df.columns else df.loc[best, "leader_final"])
            df.loc[best, "final_score"]  = min(100, base_score + 5)


# ════════════════════════════════════════════════════════════════════
# 4. 재무 통합 파이프라인 헬퍼
# ════════════════════════════════════════════════════════════════════

def _integrate_fundamentals(df: pd.DataFrame, market: str, top_n_pre: int = 120) -> pd.DataFrame:
    """
    전체 유니버스 중 상위 top_n_pre개에만 재무 수집 후
    Catalyst Score를 재무 기반으로 보정.
    """
    from screener.fundamental import fetch_fundamentals_batch

    # 1차 후보: 백테스트 검증점수 우선, 없으면 leader_final 기준
    base_col = "validated_score" if "validated_score" in df.columns else "leader_final"
    candidates = df.nlargest(top_n_pre, base_col).index.tolist()
    logger.info(f"재무 수집 대상: {len(candidates)}개")

    try:
        fund_data = fetch_fundamentals_batch(candidates, market)
    except Exception as e:
        logger.warning(f"재무 수집 실패: {e}")
        return df

    # 재무 점수를 df에 병합 후 final_score 보정
    for ticker, fd in fund_data.items():
        if ticker not in df.index: continue
        fs    = fd.get("fund_score")   # None이면 데이터 없음
        grade = fd.get("fund_grade", "")

        # fund_score가 None이면 (수집 실패) 점수 조정 없이 메타만 저장
        if fs is not None:
            # 재무 점수 → final_score에만 반영 (+/-8점). leader_final/validated_score는 고정.
            adj = (float(fs) - 50) / 50 * 8
            base_score = float(df.loc[ticker, "final_score"] if "final_score" in df.columns else df.loc[ticker, "leader_final"])
            df.loc[ticker, "final_score"] = round(min(100, max(0, base_score + adj)), 1)

        # 재무 메타 저장 (수집 실패해도 저장, fund_grade는 빈 문자열)
        df.loc[ticker, "fund_score"]        = float(fs) if fs is not None else float("nan")
        df.loc[ticker, "fund_grade"]        = grade
        df.loc[ticker, "data_available"]    = bool(fd.get("data_available", False))
        df.loc[ticker, "rev_growth"]        = fd.get("rev_growth",   float("nan"))
        df.loc[ticker, "op_growth"]         = fd.get("op_growth",    float("nan"))
        df.loc[ticker, "eps_growth"]        = fd.get("eps_growth",   float("nan"))
        df.loc[ticker, "rev_accel"]         = fd.get("rev_accel",    False)
        df.loc[ticker, "trailing_pe"]       = fd.get("trailing_pe",  float("nan"))
        df.loc[ticker, "forward_pe"]        = fd.get("forward_pe",   float("nan"))
        df.loc[ticker, "peg"]               = fd.get("peg",          float("nan"))
        df.loc[ticker, "op_margin"]         = fd.get("op_margin",    float("nan"))
        df.loc[ticker, "gross_margin"]      = fd.get("profit_margin",float("nan"))

        # 성장 가속 + 실적 좋은 종목 → Catalyst 표시값과 final_score만 보정
        if fs is not None and fd.get("rev_accel") and float(fs) >= 65:
            df.loc[ticker, "catalyst_score"] = min(100,
                float(df.loc[ticker,"catalyst_score"]) + 12)
            base_score = float(df.loc[ticker, "final_score"] if "final_score" in df.columns else df.loc[ticker, "leader_final"])
            df.loc[ticker, "final_score"] = min(100, base_score + 3)

    logger.info(f"재무 통합 완료: {len(fund_data)}개")
    return df


# ════════════════════════════════════════════════════════════════════
# 5. 4트랙 분리 (Turnaround 추가)
# ════════════════════════════════════════════════════════════════════



def _apply_context_factors(df: pd.DataFrame, factors: dict):
    """
    GPT가 제안한 팩터 가중치로 final_score 조정.
    factors: {"catalyst": 1.2, "momentum": 1.0, "supply": 1.1, ...}
    """
    if not factors or df.empty:
        return

    cat_mult = float(factors.get("catalyst", 1.0))
    mom_mult = float(factors.get("momentum", 1.0))
    vol_mult = float(factors.get("volume",   1.0))
    sup_mult = float(factors.get("supply",   1.0))
    buy_mult = float(factors.get("buyable",  1.0))

    # 팩터 가중치가 1.0에서 벗어난 만큼 final_score만 조정
    for ticker in df.index:
        try:
            adj = 0.0
            if "catalyst_score" in df.columns:
                adj += (float(df.loc[ticker,"catalyst_score"] or 50)-50) * (cat_mult-1.0) * 0.1
            if "momentum_score" in df.columns:
                adj += (float(df.loc[ticker,"momentum_score"] or 50)-50) * (mom_mult-1.0) * 0.08
            if "volume_score" in df.columns:
                adj += (float(df.loc[ticker,"volume_score"] or 50)-50)   * (vol_mult-1.0) * 0.06
            if "supply_score" in df.columns:
                adj += (float(df.loc[ticker,"supply_score"] or 50)-50)   * (sup_mult-1.0) * 0.08
            if "buyable_score" in df.columns:
                adj += (float(df.loc[ticker,"buyable_score"] or 50)-50)  * (buy_mult-1.0) * 0.06

            base_score = float(df.loc[ticker, "final_score"] if "final_score" in df.columns else df.loc[ticker, "leader_final"])
            df.loc[ticker,"final_score"]  = min(100, max(0, base_score + adj))
        except Exception:
            pass

    logger.info(f"Context 팩터 조정 완료: {factors}")


def _apply_avoid_penalty(df: pd.DataFrame, avoid_tickers: list):
    """GPT가 제안한 기피 종목에 Risk 패널티."""
    for ticker in (avoid_tickers or []):
        if ticker in df.index:
            df.loc[ticker,"top_risk_score"] = min(100, float(df.loc[ticker,"top_risk_score"] or 0)+25)
            base_score = float(df.loc[ticker, "final_score"] if "final_score" in df.columns else df.loc[ticker, "leader_final"])
            df.loc[ticker,"final_score"]    = max(0, base_score - 15)
            df.loc[ticker,"risk_flag"]      = "AI기피"
    if avoid_tickers:
        logger.info(f"Avoid 패널티: {avoid_tickers}")

def _split_tracks(df: pd.DataFrame, top_n: int) -> tuple:
    """
    4개 트랙으로 분리:
    leader_df, buyable_df, breakout_df, turnaround_df
    """
    df = df[~df["track"].isin(AVOID_TRACKS)].copy()

    # ① Hot Leader + 검증점수 상위 후보
    # 화면 정렬만 validated_score로 바꾸는 것이 아니라, 후보 풀 생성 단계부터 반영한다.
    leader_sort_col = "validated_score" if "validated_score" in df.columns else "leader_final"
    leader_df = df[df["track"].isin(LEADER_TRACKS)].nlargest(min(top_n, 20), leader_sort_col)
    if "validated_score" in df.columns:
        validated_df = df.nlargest(min(20, len(df)), "validated_score")
        leader_df = pd.concat([leader_df, validated_df], axis=0)
        leader_df = leader_df[~leader_df.index.duplicated(keep="first")]
        leader_df = leader_df.sort_values("validated_score", ascending=False).head(min(max(top_n, 20), len(leader_df)))

    # ② Breakout
    breakout_df = df[df["track"].isin(BREAKOUT_TRACKS)].nlargest(15,"breakout_score")

    # ③ Buyable
    buyable_df = df[df["buyable_score"]>=55].nlargest(top_n,"buyable_final")

    # ④ Turnaround: 52주 저점 대비 10~40% 반등 + 모멘텀 개선 중
    if "week52_low_prox" in df.columns and "catalyst_score" in df.columns:
        turn_mask = (
            df["week52_low_prox"].fillna(0).between(10, 45) &
            (df["catalyst_score"].fillna(0) >= 45) &
            (df["momentum_score"].fillna(0) >= 30)
        )
        turnaround_df = df[turn_mask].nlargest(15,"catalyst_score").copy()
        # track을 Turnaround / Early로 명시 (카드 표시 + 예상수익률 반영)
        turnaround_df["track"] = "Turnaround / Early"
        # full_df에도 turnaround_flag 저장 (예상수익률 탭에서 중복 시 올바른 track 사용)
        df.loc[turn_mask, "turnaround_flag"] = True
    else:
        turnaround_df = pd.DataFrame()

    return leader_df, buyable_df, breakout_df, turnaround_df


# ════════════════════════════════════════════════════════════════════
# 미국 파이프라인
# ════════════════════════════════════════════════════════════════════

def run_us_pipeline(
    us_data: dict, top_n: int = 30, use_cache: bool = True,
    theme_tickers: list = None, saved_themes: list = None,
    integrate_fundamentals: bool = True,
    context_factors: dict = None, avoid_tickers: list = None,
) -> tuple:
    """Returns: (leader_df, buyable_df, breakout_df, turnaround_df, excluded, full_df)"""

    if use_cache:
        cached = _load_cache("us")
        if cached is not None:
            df = cached.copy()   # 원본 캐시 보호
            if theme_tickers:    # 테마 변경 시 반드시 재적용
                _apply_theme_boost(df, theme_tickers, saved_themes)
            # Context Layer 적용
            if context_factors:
                _apply_context_factors(df, context_factors)
            if avoid_tickers:
                _apply_avoid_penalty(df, avoid_tickers)
            return (*_split_tracks(df, top_n), [], df)

    f = US_FILTER; excluded = []

    # 섹터 ETF 포함 모든 벤치마크
    bench_close = {k:v["ohlcv"]["Close"] for k,v in us_data.items()
                   if is_us_live_etf_ticker(k) and "ohlcv" in v}

    # 섹터 ETF 명시적 다운로드 (bench_close에 없는 것 추가)
    try:
        from screener.backtest import fetch_sector_etfs
        sector_etf_data = fetch_sector_etfs()
        for etf, series in sector_etf_data.items():
            if etf not in bench_close:
                bench_close[etf] = series
        logger.info(f"섹터 ETF 추가: {len(sector_etf_data)}개 → bench_close {len(bench_close)}개")
    except Exception as e:
        logger.warning(f"섹터 ETF 추가 실패: {e}")

    valid = {}
    for t,item in us_data.items():
        if is_us_live_etf_ticker(t): continue
        if (item.get("price",0) or 0)      < f["min_price"]:      excluded.append(t); continue
        if (item.get("market_cap",0) or 0) < f["min_market_cap"]: excluded.append(t); continue
        if (item.get("avg_dv",0) or 0)     < f["min_avg_dv"]:     excluded.append(t); continue
        if len(item.get("ohlcv",pd.DataFrame())) < f["min_history_days"]: excluded.append(t); continue
        item["sector"] = normalize_sector(item.get("sector",""))
        valid[t] = item

    logger.info(f"US 필터 후: {len(valid)}개")

    sect_r20={}; sect_r60={}
    for t,item in valid.items():
        c=item["ohlcv"]["Close"]; s=item.get("sector","Other")
        r20=_ret(c,20); r60=_ret(c,60)
        if math.isfinite(r20): sect_r20.setdefault(s,[]).append(r20)
        if math.isfinite(r60): sect_r60.setdefault(s,[]).append(r60)
    sect_avg20={s:float(np.mean(v)) for s,v in sect_r20.items()}
    sect_avg60={s:float(np.mean(v)) for s,v in sect_r60.items()}

    spy_r=float((bench_close["SPY"].iloc[-1]/bench_close["SPY"].iloc[-21]-1)*100) \
          if "SPY" in bench_close and len(bench_close["SPY"])>20 else 0
    tickers_list=list(valid.keys())
    rs_ranks=_pct_rank([_ret(valid[t]["ohlcv"]["Close"],20)-spy_r for t in tickers_list])

    regime = market_regime_score(bench_close)
    logger.info(f"US Regime: {regime['us_regime']:.0f}")

    rows=[]
    for i,(t,item) in enumerate(valid.items()):
        s=item.get("sector",""); sb=SECTOR_ETF.get(s,"SPY")
        try:
            row=calc_us_factors(t,item,bench_close,sb,
                                sect_avg20.get(s,float("nan")),
                                sect_avg60.get(s,float("nan")),
                                rs_rank_pct=rs_ranks[i], regime=regime)
            rows.append(row)
        except Exception as e:
            logger.debug(f"US factor({t}): {e}"); excluded.append(t)

    if not rows:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(), excluded, pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")

    # ── 재무 통합 (파이프라인 안에서) ─────────────────────────────
    if integrate_fundamentals:
        # validated_score 기준 상위 종목을 재무 수집 대상으로 (없으면 leader_final 기준)
        if "validated_score" in df.columns:
            df = df.sort_values("validated_score", ascending=False)
        df = _integrate_fundamentals(df, "US", top_n_pre=min(40, len(df)))

    if theme_tickers: _apply_theme_boost(df, theme_tickers, saved_themes)
    if context_factors: _apply_context_factors(df, context_factors)
    if avoid_tickers:   _apply_avoid_penalty(df, avoid_tickers)
    _save_cache(df, "us")
    df.attrs["regime"] = regime

    leader_df,buyable_df,breakout_df,turnaround_df = _split_tracks(df, top_n)
    for _d in (leader_df, buyable_df, breakout_df, turnaround_df):
        try: _d.attrs["regime"] = regime
        except Exception: pass
    return leader_df, buyable_df, breakout_df, turnaround_df, excluded, df


# ════════════════════════════════════════════════════════════════════
# 한국 파이프라인
# ════════════════════════════════════════════════════════════════════

def run_kr_pipeline(
    kr_data: dict, kr_bench: dict, top_n: int = 30, use_cache: bool = True,
    theme_tickers: list = None, saved_themes: list = None,
    integrate_fundamentals: bool = True,
    context_factors: dict = None, avoid_tickers: list = None,
) -> tuple:
    """Returns: (leader_df, buyable_df, breakout_df, turnaround_df, excluded, full_df)"""

    if use_cache:
        cached = _load_cache("kr")
        if cached is not None:
            df = cached.copy()
            if theme_tickers:
                _apply_theme_boost(df, theme_tickers, saved_themes)
            if context_factors:
                _apply_context_factors(df, context_factors)
            if avoid_tickers:
                _apply_avoid_penalty(df, avoid_tickers)
            return (*_split_tracks(df, top_n), [], df)

    f=KR_FILTER; excluded=[]
    warned_set = _fetch_warned_stocks()
    sector_map = _fetch_kr_sector_map()

    valid={}
    for t,item in kr_data.items():
        if is_kr_preferred_ticker(t, f["exclude_preferred"]):
            excluded.append(t); continue
        # ETF 이름 기반 제외
        item_name = str(item.get("name","") or "")
        if is_kr_pipeline_etf_name(item_name):
            excluded.append(t); continue
        if t in warned_set or item.get("is_warned",False): excluded.append(t); continue
        if (item.get("market_cap",0) or 0) < f["min_market_cap"]: excluded.append(t); continue
        if (item.get("avg_amount",0) or 0) < f["min_avg_amount"]:  excluded.append(t); continue
        if len(item.get("ohlcv",pd.DataFrame())) < f["min_history_days"]: excluded.append(t); continue
        item["sector"] = sector_map.get(t) or _get_kr_sector(t,item.get("sector",""))
        valid[t]=item

    logger.info(f"KR 필터 후: {len(valid)}개")

    sect_r20={}
    for t,item in valid.items():
        c=item["ohlcv"]["Close"]; s=item.get("sector","기타"); r20=_ret(c,20)
        if math.isfinite(r20): sect_r20.setdefault(s,[]).append(r20)
    sect_avg20={s:float(np.mean(v)) for s,v in sect_r20.items()}

    tickers_list=list(valid.keys()); rs20_vals=[]
    for t in tickers_list:
        item=valid[t]; c=item["ohlcv"]["Close"]
        bk="kospi" if item.get("market","KOSPI")=="KOSPI" else "kosdaq"
        bench_s=kr_bench.get(bk)
        bench_r=float((bench_s.iloc[-1]/bench_s.iloc[-21]-1)*100) \
                if bench_s is not None and len(bench_s)>20 else 0
        rs20_vals.append(_ret(c,20)-bench_r)
    rs_ranks=_pct_rank(rs20_vals)

    regime=market_regime_score(kr_bench)

    rows=[]
    for i,(t,item) in enumerate(valid.items()):
        s=item.get("sector","기타")
        try:
            row=calc_kr_factors(t,item,kr_bench,sect_avg20.get(s,float("nan")),
                                rs_rank_pct=rs_ranks[i], regime=regime)
            rows.append(row)
        except Exception as e:
            logger.debug(f"KR factor({t}): {e}"); excluded.append(t)

    if not rows:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(), excluded, pd.DataFrame()

    df=pd.DataFrame(rows).set_index("ticker")

    if integrate_fundamentals:
        df = _integrate_fundamentals(df, "KR", top_n_pre=min(30, len(df)))

    if theme_tickers: _apply_theme_boost(df, theme_tickers, saved_themes)
    if context_factors: _apply_context_factors(df, context_factors)
    if avoid_tickers:   _apply_avoid_penalty(df, avoid_tickers)
    _save_cache(df, "kr")
    df.attrs["regime"] = regime

    leader_df,buyable_df,breakout_df,turnaround_df = _split_tracks(df, top_n)
    for _d in (leader_df, buyable_df, breakout_df, turnaround_df):
        try: _d.attrs["regime"] = regime
        except Exception: pass
    return leader_df, buyable_df, breakout_df, turnaround_df, excluded, df
