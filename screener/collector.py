"""
screener/collector.py  v6.1
- 미국: yfinance 배치 다운로드 (실패율 대폭 감소)
- 한국: 네이버 금융
  유니버스: KOSPI200(시총상위200) + KOSDAQ150(시총상위150) + 거래대금상위150
  OHLCV: 네이버 일봉
  수급: 네이버 외국인/기관
"""

import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

from screener.paths import (
    CACHE_DIR, TODAY, set_data_date, data_date_key,
)
LOOKBACK_CAL = 900
PERIOD_DAYS  = 520
# 네이버 일봉은 페이지당 약 10거래일입니다.
# 한국 백테스트 날짜 수를 30~60회 이상 확보하기 위해 약 2년치(52페이지/520거래일)를 수집합니다.
KR_OHLCV_PAGES = 52
MAX_WORKERS  = 8  # 페이지 수 증가에 맞춰 네이버 요청 폭주 방지

# ── 네이버 공통 세션 ─────────────────────────────────────────────────
_NAV_SESS: Optional[requests.Session] = None

def _nav() -> requests.Session:
    global _NAV_SESS
    if _NAV_SESS is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer":         "https://finance.naver.com/",
            "Accept-Language": "ko-KR,ko;q=0.9",
        })
        _NAV_SESS = s
    return _NAV_SESS


# ── 네트워크 재시도 헬퍼 ──────────────────────────────────────────────
# 네이버/외부 사이트가 일시적으로 흔들릴 때(타임아웃, 5xx, 연결 끊김) 종목이
# 통째로 누락되는 것을 막기 위한 지수 백오프 재시도.
def _get_with_retry(
    session: requests.Session,
    url: str,
    *,
    retries: int = 3,
    backoff: float = 0.6,
    **kwargs,
) -> Optional[requests.Response]:
    """성공 시 Response, 모든 시도 실패 시 None. timeout 미지정 시 10초."""
    kwargs.setdefault("timeout", 10)
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(url, **kwargs)
            if r.status_code == 200:
                return r
            last_err = f"HTTP {r.status_code}"
            # 4xx는 재시도해도 무의미 → 즉시 중단 (429 제외)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                break
        except requests.RequestException as e:
            last_err = repr(e)
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
    logger.warning(f"요청 실패({retries}회): {url} — {last_err}")
    return None


# ════════════════════════════════════════════════════════════════════
# 미국 — yfinance 배치 다운로드
# ════════════════════════════════════════════════════════════════════

def get_us_tickers() -> list[str]:
    # ── 당일 캐시 확인 (Wikipedia 크롤링은 하루 1번만) ──────────
    cache_p = CACHE_DIR / f"us_tickers_{data_date_key()}.pkl"
    if cache_p.exists():
        try:
            import pickle
            with open(cache_p, "rb") as f_:
                result = pickle.load(f_)
            logger.info(f"미국 티커 캐시 로드: {len(result)}개")
            return result
        except Exception:
            cache_p.unlink(missing_ok=True)

    tickers: set[str] = set()

    # ── S&P500 현재 구성종목 (Wikipedia) ─────────────────────────
    sp500_ok = False
    try:
        # Wikipedia는 User-Agent 없는 요청을 차단(403)하므로 헤더 필수.
        # 과거 pd.read_html(url) 직접 호출은 헤더가 없어 실패 → fallback으로 떨어졌음.
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        sp_tables = pd.read_html(StringIO(r.text))
        sp_tickers = sp_tables[0]  # 첫 번째 테이블 = 현재 구성종목
        for col in sp_tickers.columns:
            if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                raw = [str(t).replace(".", "-").strip().upper()
                       for t in sp_tickers[col].dropna()
                       if isinstance(t, str) and 0 < len(str(t)) < 7
                       and str(t).replace("-","").replace(".","").isalpha()]
                if len(raw) > 400:  # S&P500은 500개 근처여야 함
                    tickers.update(raw)
                    sp500_ok = True
                    logger.info(f"S&P500 현재 구성종목: {len(raw)}개")
                break
    except Exception as e:
        logger.warning(f"S&P500 Wikipedia: {e}")

    # ── Nasdaq-100 현재 구성종목 ──────────────────────────────────
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        tables = pd.read_html(StringIO(r.text))
        for tbl in tables:
            for col in tbl.columns:
                if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                    nq = [str(t).replace(".", "-").strip().upper()
                          for t in tbl[col].dropna()
                          if isinstance(t, str) and 0 < len(str(t)) < 8
                          and str(t).replace("-","").replace(".","").isalpha()]
                    if len(nq) > 80:  # Nasdaq-100은 100개 근처
                        tickers.update(nq)
                        logger.info(f"Nasdaq-100: {len(nq)}개")
                        break
    except Exception as e:
        logger.warning(f"Nasdaq100: {e}")

    if not sp500_ok or len(tickers) < 100:
        try:
            from data.us_universe import SP500_FALLBACK
            tickers.update(SP500_FALLBACK)
            logger.info("SP500_FALLBACK 사용")
        except Exception:
            pass

    from screener.universe import US_BENCH, UNIVERSE_ETFS
    tickers.update(US_BENCH.values())
    tickers.update(UNIVERSE_ETFS)
    # 강제 포함 유니버스 (MU, SNDK 등 핵심 테마 종목)
    try:
        from data.us_universe import FORCED_UNIVERSE
        tickers.update(FORCED_UNIVERSE)
        logger.info(f"강제 유니버스 {len(FORCED_UNIVERSE)}개 추가")
    except Exception:
        pass

    # 동적 유니버스 (거래대금 급증 + 52주 신고가 — 중형 성장주 포착)
    # 캐시: 하루 1번만 스캔
    dyn_cache = CACHE_DIR / f"dynamic_universe_{data_date_key()}.pkl"
    if dyn_cache.exists():
        try:
            with open(dyn_cache,"rb") as f: dynamic_tickers = pickle.load(f)
            logger.info(f"동적 유니버스 캐시: {len(dynamic_tickers)}개")
        except Exception: dynamic_tickers = []
    else:
        dynamic_tickers = _fetch_us_momentum_universe(n=60)
        try:
            with open(dyn_cache,"wb") as f: pickle.dump(dynamic_tickers,f)
        except Exception: pass

    if dynamic_tickers:
        tickers.update(dynamic_tickers)
        logger.info(f"동적 유니버스 {len(dynamic_tickers)}개 추가")

    # ── 죽은 티커 제거 (상폐/합병/개명으로 yfinance에 없는 것) ──────
    # 매 수집마다 실패 로그를 더럽히고 재시도 시간을 잡아먹어 미리 제거.
    # 명백히 상폐·합병·개명된 것만 보수적으로 등재.
    DEAD_TICKERS = {
        "STRZA","WCRX","BMC","VIP","VMED","PPDI","KRFT","KFT","HANS","FWLT",
        "GMCR","DISH","VIAB","WLTW","ALXN","LEAP","WBA","MYL","WFMI","CEPH",
        "CERN","ANSS","CMCSK","CTRP","CTRX","DTV","ENDP","FMCN","FLIR","LMCA",
        "MXIM","NUAN","SGEN","SRCL","SHPG","SPLK","UAUA","DISCA","CA","FB",
        "CREE","WLTW","XLNX","MNTA","CXO","ATVI","ABMD","ANTM","RTN","CTXS",
        # 추가: 수집 로그에 반복 등장한 상폐/합병/개명 티커
        # (CYBR·ZI·BLDE·HOLX·CFLT는 살아있어 제외 — 일시적 yfinance 오류일 뿐)
        "NEP","FNSR","IIVI","ACIA","NAND","VIAVI","STEC","LILM","MNTV",
        "SCWX","AMER","BASE","ALTR","HCP",
    }
    before = len(tickers)
    tickers = {t for t in tickers if t not in DEAD_TICKERS}
    removed = before - len(tickers)
    if removed:
        logger.info(f"죽은 티커 {removed}개 제거")

    result = sorted(tickers)
    logger.info(f"미국 유니버스: {len(result)}개")

    # 캐시 저장
    try:
        import pickle
        with open(cache_p, "wb") as f_:
            pickle.dump(result, f_)
    except Exception:
        pass

    return result



def _fetch_us_momentum_universe(n: int = 60) -> list[str]:
    """
    동적 유니버스: 기존 유니버스에서 최근 모멘텀이 강한 종목 추출.

    레이어 구성:
    A. 기본 S&P500/Nasdaq100 (정적)
    B. FORCED_UNIVERSE 테마 종목 (정적)
    C. 거래대금 급증 + 52주 신고가 근접 종목 (동적 ← 이 함수)
    D. 최근 수익률 상위 종목 (동적)

    전체 yfinance 스캔이 불가능하므로, 알려진 중형 성장주 후보군 내에서
    실제 가격 데이터로 모멘텀을 검증합니다.
    """
    import yfinance as yf

    # 동적 스캔 대상 후보 (S&P500/Nasdaq100 외 중형 성장주)
    SCAN_CANDIDATES = [
        # 반도체 장비/소재
        "AMAT","LRCX","KLAC","TER","COHR","MKSI","ENTG","ONTO","ACLS","ICHR",
        "AMKR","FORM","MTSI","ALGM","WOLF","CREE","OSIS",
        # AI/데이터센터 인프라
        "SMCI","NTAP","PURE","PSTG","SOUN","BBAI","GTLB","ESTC","DOCN","DOMO",
        "NCNO","ALTR","CFLT","DT","FROG","HCP","IOT","MNTV","TOST","VG",
        # 메모리/스토리지 확장
        "NAND","STEC","PRGS","IIVI","AAOI","ACIA","VIAVI","LITE","FNSR",
        # 전력/에너지 인프라
        "MYRG","WATT","GRID","ARRY","HASI","CWEN","NEP","ORA","BEP","CLNE",
        # 사이버보안 중형
        "TENB","VRNS","QLYS","SAIL","RPD","STNE","AMER","SCWX","DCI",
        # 방산/항공 중형
        "KTOS","AVAV","RCAT","JOBY","ACHR","LILM","EVEX","BLDE",
        # 바이오 성장주
        "RXRX","SMMT","IOVA","ACAD","NTRA","EXAS","INVA","PRAX","PTGX",
        # 핀테크/결제
        "AFRM","UPST","MQ","FLYW","TASK","RELY","PAYC","HHH",
        # 리테일/소비 성장
        "DUOL","HIMS","MODV","XPOF","BURL","FIVE","OLLI",
        # 클라우드/SaaS
        "BILL","ZI","SEMR","ASAN","BASE","CLSK","BRZE","AMPL","ALRM",
    ]

    try:
        # 배치로 최근 60일 데이터 다운로드
        raw = yf.download(
            SCAN_CANDIDATES,
            period="3mo",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if raw.empty:
            logger.warning("동적 유니버스 스캔 실패 (데이터 없음)")
            return []

        results = []
        for ticker in SCAN_CANDIDATES:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    close  = raw[ticker]["Close"].dropna()
                    volume = raw[ticker]["Volume"].dropna() if "Volume" in raw[ticker] else pd.Series()
                    high   = raw[ticker]["High"].dropna()   if "High"   in raw[ticker] else pd.Series()
                else:
                    close = raw["Close"].dropna()
                    volume= raw["Volume"].dropna() if "Volume" in raw else pd.Series()
                    high  = raw["High"].dropna()   if "High"   in raw else pd.Series()

                if len(close) < 20: continue

                # 지표 계산
                ret_20d   = (close.iloc[-1]/close.iloc[-21]-1)*100 if len(close)>=21 else float("nan")
                ret_5d    = (close.iloc[-1]/close.iloc[-6]-1)*100  if len(close)>=6  else float("nan")

                # 52주 신고가 근접도
                prox_52w  = float("nan")
                if len(high) >= 5:
                    h_max = high.tail(min(252,len(high))).max()
                    if h_max > 0: prox_52w = (close.iloc[-1]/h_max)*100

                # 거래대금 급증 (최근 5일 vs 이전 20일 평균)
                vol_surge = float("nan")
                if len(volume) >= 25:
                    avg_vol_20 = float(volume.tail(25).head(20).mean())
                    avg_vol_5  = float(volume.tail(5).mean())
                    if avg_vol_20 > 0: vol_surge = avg_vol_5/avg_vol_20

                # 모멘텀 점수 계산 (단순)
                score = 0
                if math.isfinite(ret_20d):
                    if ret_20d >= 20: score += 30
                    elif ret_20d >= 10: score += 20
                    elif ret_20d >= 5: score += 10
                if math.isfinite(ret_5d):
                    if ret_5d >= 5: score += 20
                    elif ret_5d >= 2: score += 10
                if math.isfinite(prox_52w):
                    if prox_52w >= 95: score += 25   # 신고가 근처
                    elif prox_52w >= 85: score += 15
                if math.isfinite(vol_surge):
                    if vol_surge >= 2.0: score += 25  # 거래량 2배 이상
                    elif vol_surge >= 1.5: score += 15

                if score >= 40:  # 임계값 이상만 포함
                    results.append((ticker, score))

            except Exception:
                continue

        # 점수 상위 N개
        results.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t,_ in results[:n]]
        if selected:
            logger.info(f"동적 유니버스: {len(selected)}개 발굴 (거래대금급증+신고가 기준)")
        return selected

    except Exception as e:
        logger.warning(f"동적 유니버스 스캔 오류: {e}")
        return []


def get_dynamic_universe_hints() -> list[str]:
    """
    Russell 1000 / Nasdaq 전체 중 거래대금 상위를 가져오려면
    별도 API가 필요하므로, 현재는 curated 목록으로 대체.
    다음 티커들을 동적 유니버스 힌트로 제공:
    - 최근 AI/데이터센터 공급망 핵심
    - 에너지/전력 인프라
    - 방산/사이버보안 중형주
    """
    return [
        # AI 인프라 중형주
        "SMCI","DELL","HPE","NTAP","PURE","PSTG","AI","SOUN","BBAI",
        # 반도체 장비/소재
        "AMAT","LRCX","KLAC","ASML","TER","COHR","MKSI","ENTG","ONTO",
        # AI 스토리지/메모리
        "MU","SNDK","WDC","STX","SIMO","MRVL","QRVO",
        # 전력 인프라
        "VST","CEG","NRG","ETR","GEV","PWR","WATT","EME","MYRG",
        # 사이버보안 중형
        "CRWD","PANW","ZS","S","CYBR","TENB","VRNS","QLYS",
        # 방산 중형
        "AXON","CACI","SAIC","LDOS","KTOS","AVAV","HII",
        # 바이오/제약 중형
        "RXRX","ALNY","SMMT","IOVA","ACAD","NTRA","EXAS",
        # 데이터/클라우드 인프라
        "DDOG","NET","ZS","SNOW","MDB","GTLB","ESTC",
    ]

def fetch_us_universe(
    tickers: list[str],
    progress_cb: Optional[Callable] = None,
) -> tuple[dict, list[str]]:
    """
    yfinance.download() 배치로 OHLCV 수집 → 실패율 대폭 감소.
    메타(sector, market_cap 등)는 개별 Ticker.info로 별도 수집.
    """
    import yfinance as yf

    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_CAL)

    # ── 1단계: 배치 OHLCV 병렬 다운로드 ────────────────────────
    # 레이트리밋/타임아웃 대응: 배치를 작게(50) + 동시수 낮게(2) + 배치간 지연.
    # 대량(150×4=600 동시요청) 시 Yahoo가 타임아웃을 내며 멀쩡한 종목까지
    # 무더기로 실패하는 문제가 있어 보수적으로 조정.
    import time as _time
    BATCH        = 50    # 배치 크기 (과거 150 → 타임아웃 유발)
    PARALLEL     = 2     # 동시 배치 수 (과거 4 → 레이트리밋 유발)
    all_ohlcv: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i+BATCH] for i in range(0, len(tickers), BATCH)]
    total_batches = len(batches)

    def _parse_raw(raw, batch) -> dict:
        """yf.download 결과를 종목별 DataFrame dict로 변환."""
        result = {}
        if raw is None or raw.empty:
            return result
        if isinstance(raw.columns, pd.MultiIndex):
            for t in batch:
                try:
                    df = raw[t].dropna(how="all")
                    if len(df) >= 20:
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        df = df.rename(columns=str.capitalize).tail(PERIOD_DAYS)
                        if "Close" in df.columns:
                            df["Amount"] = df["Close"] * df.get("Volume", 0)
                            result[t] = df
                except Exception:
                    pass
        else:
            t = batch[0]
            df = raw.dropna(how="all")
            if len(df) >= 20:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.rename(columns=str.capitalize).tail(PERIOD_DAYS)
                if "Close" in df.columns:
                    df["Amount"] = df["Close"] * df.get("Volume", 0)
                    result[t] = df
        return result

    def _dl(batch, use_threads=True):
        """단일 배치 다운로드 (예외 시 빈 dict).
        use_threads=False는 yfinance의 'dictionary changed size' 버그 회피용."""
        try:
            raw = yf.download(
                batch, start=start, end=end,
                auto_adjust=True, actions=False,
                progress=False, threads=use_threads,
                group_by="ticker",
            )
            return _parse_raw(raw, batch)
        except Exception as e:
            # yfinance 동시성 버그(dictionary changed size 등)면 단일스레드로 즉시 재시도
            if use_threads and ("changed size" in str(e) or "iteration" in str(e)):
                _time.sleep(0.3)
                return _dl(batch, use_threads=False)
            logger.warning(f"배치 다운로드 실패({len(batch)}개): {e}")
            return {}

    def _download_batch(args):
        bi, batch = args
        res = _dl(batch)
        # 배치 절반 이상 누락이면 → 빠진 것만 단일스레드로 '딱 한 번' 재시도.
        # (타임아웃으로 멀쩡한 종목이 통째로 날아가는 것만 방지. 더는 매달리지 않음)
        if len(res) < len(batch) * 0.5:
            missing = [t for t in batch if t not in res]
            _time.sleep(0.4)
            res.update(_dl(missing, use_threads=False))
        return res

    done_count = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        fmap = {ex.submit(_download_batch, (bi, batch)): bi
                for bi, batch in enumerate(batches)}
        for fut in as_completed(fmap):
            bi = fmap[fut]
            done_count += len(batches[bi])
            if progress_cb:
                progress_cb(done_count, len(tickers), f"배치 {bi+1}/{total_batches}")
            try:
                all_ohlcv.update(fut.result())
            except Exception as e:
                logger.warning(f"배치 결과 처리 실패({bi}): {e}")

    logger.info(f"배치 OHLCV 1차 수집: {len(all_ohlcv)}개 성공")

    # ── 1.5단계: 여전히 빠진 종목 '딱 한 번' 더 재시도 후 버림 ──────
    still_missing = [t for t in tickers if t not in all_ohlcv]
    if still_missing and len(still_missing) < len(tickers):
        n_miss = len(still_missing)
        logger.info(f"미수집 {n_miss}개 최종 재시도(1회)")
        for j in range(0, n_miss, 30):
            sub = still_missing[j:j+30]
            if progress_cb:
                progress_cb(len(tickers), len(tickers),
                            f"미수집 재시도 {min(j+30,n_miss)}/{n_miss}")
            try:
                all_ohlcv.update(_dl(sub, use_threads=False))
            except Exception:
                pass
        # 여기까지 안 잡힌 종목은 깔끔히 포기 (상폐/먹통 티커)
        logger.info(f"재시도 후 누적: {len(all_ohlcv)}개")

    logger.info(f"배치 OHLCV 최종 수집: {len(all_ohlcv)}개 성공")

    # ── 2단계: 메타 정보 개별 수집 (병렬) ────────────────────────
    def fetch_meta(ticker: str) -> Optional[dict]:
        try:
            obj  = yf.Ticker(ticker)
            info = obj.info or {}
            earnings_date = None
            try:
                cal = obj.calendar
                if cal is not None and not cal.empty:
                    ec = [c for c in cal.columns if "earnings" in str(c).lower()]
                    if ec:
                        earnings_date = str(cal[ec[0]].iloc[0])[:10]
            except Exception:
                pass
            return {
                "company":       info.get("shortName") or info.get("longName") or ticker,
                "sector":        info.get("sector", ""),
                "industry":      info.get("industry", ""),
                "market_cap":    info.get("marketCap", np.nan),
                "earnings_date": earnings_date,
            }
        except Exception:
            return {"company": ticker, "sector": "", "industry": "",
                    "market_cap": np.nan, "earnings_date": None}

    # 배치에서 성공한 종목만 메타 수집
    meta_tickers = list(all_ohlcv.keys())
    metas: dict  = {}
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fmap = {ex.submit(fetch_meta, t): t for t in meta_tickers}
        for fut in as_completed(fmap):
            t = fmap[fut]
            done += 1
            if progress_cb:
                progress_cb(done, len(meta_tickers), f"메타 {t}")
            try:
                metas[t] = fut.result() or {}
            except Exception:
                metas[t] = {}

    # ── 조합 ────────────────────────────────────────────────────
    data:   dict = {}
    failed: list = []

    for t in tickers:
        if t not in all_ohlcv:
            failed.append(t)
            continue
        df   = all_ohlcv[t]
        meta = metas.get(t, {})
        dv   = (df["Close"] * df.get("Volume", 0)).tail(20).mean()
        data[t] = {
            "ticker":        t,
            "ohlcv":         df,
            "company":       meta.get("company", t),
            "sector":        meta.get("sector", ""),
            "industry":      meta.get("industry", ""),
            "market_cap":    meta.get("market_cap", np.nan),
            "avg_dv":        float(dv) if np.isfinite(dv) else np.nan,
            "price":         float(df["Close"].iloc[-1]),
            "earnings_date": meta.get("earnings_date"),
        }

    logger.info(f"미국 최종: 성공 {len(data)}, 실패 {len(failed)}")
    if data:
        set_data_date(only_if_absent=True)   # 이 데이터셋의 기준일 기록(최초 1회)
    return data, failed


def fetch_single_us_ticker_ondemand(
    ticker: str,
) -> "tuple[dict | None, dict[str, pd.Series], str]":
    """
    단일 미국 티커 on-demand 수집 (yfinance).
    스크리너 배치 수집과 무관하게 즉시 조회한다.
    스크리너 로드 여부와 관계없이 동작하며, 실패해도 기존 결과에 영향 없음.

    Returns:
        (item, bench_close, error_msg)
        - item: calc_us_factors에 넘길 dict. 실패 시 None.
        - bench_close: {"SPY": Series, "QQQ": Series, …}
        - error_msg: 성공 시 "", 실패 시 설명 문자열
    """
    import math as _math
    import yfinance as yf

    t = str(ticker).strip().upper()
    if not t:
        return None, {}, "티커가 비어 있습니다."

    BENCH_TICKERS = [
        "SPY", "QQQ",
        "XLK", "XLF", "XLV", "XLI", "XLE", "XLRE", "XLY", "XLC", "XLB", "XLU",
        "SMH",
    ]
    all_tickers = [t] + [b for b in BENCH_TICKERS if b != t]

    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_CAL)

    try:
        raw = yf.download(
            all_tickers,
            start=start, end=end,
            auto_adjust=True, actions=False,
            progress=False, threads=False,
            group_by="ticker",
        )
    except Exception as exc:
        return None, {}, f"yfinance 다운로드 실패: {exc}"

    if raw is None or raw.empty:
        return None, {}, f"{t}: Yahoo Finance 응답이 비어 있습니다."

    def _extract(sym: str) -> Optional[pd.DataFrame]:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[sym].dropna(how="all")
            else:
                df = raw.dropna(how="all")
            if len(df) < 20:
                return None
            df = df.copy()
            # tz-aware이면 UTC→naive, tz-naive이면 그대로 유지
            idx = pd.to_datetime(df.index, utc=True).tz_convert(None)
            df.index = idx
            df = df.rename(columns=str.capitalize).tail(PERIOD_DAYS)
            return df if "Close" in df.columns else None
        except Exception:
            return None

    # 벤치마크 Close Series 구성
    bench_close: dict[str, pd.Series] = {}
    for bt in BENCH_TICKERS:
        df_b = _extract(bt)
        if df_b is not None:
            bench_close[bt] = df_b["Close"]

    # 타겟 티커 OHLCV
    df_t = _extract(t)
    if df_t is None:
        return None, bench_close, (
            f"{t}: OHLCV 데이터 없음 — 존재하지 않는 티커이거나 Yahoo Finance 수집 실패"
        )

    df_t = df_t.copy()
    vol_col = df_t.get("Volume", pd.Series(0, index=df_t.index))
    df_t["Amount"] = df_t["Close"] * vol_col

    # 메타 정보 (개별 Ticker.info)
    company      = t
    sector       = ""
    industry     = ""
    market_cap   = np.nan
    earnings_date = None
    try:
        obj  = yf.Ticker(t)
        info = obj.info or {}
        company    = info.get("shortName") or info.get("longName") or t
        sector     = info.get("sector", "")
        industry   = info.get("industry", "")
        market_cap = info.get("marketCap", np.nan)
        try:
            cal = obj.calendar
            if cal is not None and not cal.empty:
                ec = [c for c in cal.columns if "earnings" in str(c).lower()]
                if ec:
                    earnings_date = str(cal[ec[0]].iloc[0])[:10]
        except Exception:
            pass
    except Exception:
        pass

    dv = float(df_t["Amount"].tail(20).mean()) if "Amount" in df_t.columns else np.nan

    try:
        mc_f = float(market_cap)
        mc_out = mc_f if _math.isfinite(mc_f) else np.nan
    except (TypeError, ValueError):
        mc_out = np.nan

    item = {
        "ticker":        t,
        "ohlcv":         df_t,
        "company":       company,
        "sector":        sector,
        "industry":      industry,
        "market_cap":    mc_out,
        "avg_dv":        float(dv) if np.isfinite(dv) else np.nan,
        "price":         float(df_t["Close"].iloc[-1]),
        "earnings_date": earnings_date,
    }

    logger.info(f"on-demand 수집 완료: {t} ({company}) | {len(df_t)}일 | 벤치 {len(bench_close)}개")
    return item, bench_close, ""


# ════════════════════════════════════════════════════════════════════
# 한국 — 네이버 금융 (제한된 유니버스)
# ════════════════════════════════════════════════════════════════════

def _naver_marcap_top(market: str, max_n: int) -> list[tuple[str, str]]:
    """
    네이버 시가총액 순 상위 N개 수집.
    market: 'KOSPI'(sosok=0) or 'KOSDAQ'(sosok=1)
    """
    sosok = "0" if market == "KOSPI" else "1"
    url   = "https://finance.naver.com/sise/sise_market_sum.naver"
    s     = _nav()
    codes: dict[str, str] = {}

    page = 1
    while len(codes) < max_n:
        try:
            r    = _get_with_retry(s, url, params={"sosok": sosok, "page": page})
            if r is None: break
            soup = BeautifulSoup(r.text, "html.parser")
            found = 0
            for a in soup.select("a[href*='code=']"):
                code = a["href"].split("code=")[1][:6]
                name = a.text.strip()
                if code.isdigit() and len(code) == 6 and name and code not in codes:
                    codes[code] = name
                    found += 1
            if found == 0:
                break
            # 마지막 페이지 체크
            pager = soup.select("td.pgRR > a")
            if not pager:
                break
            page += 1
            time.sleep(0.12)
        except Exception as e:
            logger.debug(f"네이버 시총 {market} p{page}: {e}")
            break

    result = list(codes.items())[:max_n]
    logger.info(f"네이버 {market} 시총 상위: {len(result)}개")
    return result


def _naver_volume_top(max_n: int) -> list[tuple[str, str]]:
    """
    #5 수정: KOSPI / KOSDAQ 각각 독립적으로 max_n//2개씩 수집 후 합산.
    기존 구조는 KOSPI에서 max_n이 채워지면 KOSDAQ이 무의미해지는 버그.
    """
    s   = _nav()
    url = "https://finance.naver.com/sise/sise_quant.naver"

    per_market = max(max_n // 2, 50)   # 각 시장에서 최소 50개

    def _collect(sosok: str, limit: int) -> dict[str, str]:
        codes: dict[str, str] = {}
        page = 1
        while len(codes) < limit:
            try:
                r    = _get_with_retry(s, url, params={"sosok": sosok, "page": page})
                if r is None: break
                soup = BeautifulSoup(r.text, "html.parser")
                found = 0
                for a in soup.select("a[href*='code=']"):
                    code = a["href"].split("code=")[1][:6]
                    name = a.text.strip()
                    if code.isdigit() and len(code) == 6 and name and code not in codes:
                        codes[code] = name
                        found += 1
                if found == 0:
                    break
                pager = soup.select("td.pgRR > a")
                if not pager:
                    break
                page += 1
                time.sleep(0.10)
            except Exception as e:
                logger.debug(f"거래대금 sosok={sosok} p{page}: {e}")
                break
        return codes

    kospi_codes  = _collect("0", per_market)
    kosdaq_codes = _collect("1", per_market)

    # 합산 (KOSDAQ이 겹치지 않게)
    merged: dict[str, str] = {**kospi_codes}
    for k, v in kosdaq_codes.items():
        merged.setdefault(k, v)

    result = list(merged.items())[:max_n]
    logger.info(f"거래대금 상위: KOSPI {len(kospi_codes)} + KOSDAQ {len(kosdaq_codes)} → 합산 {len(result)}개")
    return result


def get_kr_tickers() -> list[str]:
    """
    KOSPI200 + KOSDAQ150 + 거래대금상위150 + 테마 강제 유니버스.
    중복 제거 후 약 350~450개.

    레이어 구성:
    1. KOSPI 시총 상위 200 (네이버)
    2. KOSDAQ 시총 상위 150 (네이버)
    3. 거래대금 상위 150 (당일 모멘텀 포착)
    4. 테마 강제 유니버스 (HBM/AI, 방산, 전력, 배터리, 바이오, 원전, 로봇)
    """
    codes: dict[str, str] = {}

    for code, name in _naver_marcap_top("KOSPI", 200):
        codes[code] = name
    for code, name in _naver_marcap_top("KOSDAQ", 150):
        codes[code] = name
    for code, name in _naver_volume_top(150):
        codes.setdefault(code, name)

    # 테마 강제 유니버스 (S&P500의 FORCED_UNIVERSE 역할)
    try:
        from data.kr_universe import FORCED_KR_UNIVERSE
        before = len(codes)
        for code, name in FORCED_KR_UNIVERSE.items():
            codes.setdefault(code, name)
        added = len(codes) - before
        if added > 0:
            logger.info(f"한국 테마 강제 유니버스 {added}개 추가")
    except Exception as e:
        logger.warning(f"한국 강제 유니버스 로드 실패: {e}")

    logger.info(f"한국 유니버스 최종: {len(codes)}개 (KOSPI200+KOSDAQ150+거래대금+테마)")
    return sorted(codes.keys())


def _market_of(code: str, kospi_set: set, kosdaq_set: set) -> str:
    if code in kospi_set:   return "KOSPI"
    if code in kosdaq_set:  return "KOSDAQ"
    return "KOSPI"


# ── 한국 OHLCV (네이버 일봉) ────────────────────────────────────────

def _naver_ohlcv(code: str, pages: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    네이버 일봉 OHLCV.
    - 기존 6페이지는 약 60거래일이라 KR 백테스트에 필요한 85~205거래일을 충족하지 못했습니다.
    - 기본 52페이지를 읽고 약 520거래일로 잘라 사용합니다.
    - 같은 날 같은 종목은 캐시를 사용해 반복 실행 시간을 줄입니다.
    """
    pages = pages or KR_OHLCV_PAGES
    cache_p = CACHE_DIR / f"kr_ohlcv_{code}_{pages}p.pkl"  # 날짜 제거: 영구 보존(스탬프로 기준일 추적)
    if cache_p.exists():
        try:
            with open(cache_p, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached
        except Exception:
            cache_p.unlink(missing_ok=True)

    s   = _nav()
    url = "https://finance.naver.com/item/sise_day.naver"
    all_rows = []

    for page in range(1, pages + 1):
        try:
            # 핫패스: 종목당 수십 페이지 순차 호출 → 페이지 1개 실패는 비치명적.
            # 재시도/백오프를 1회로 제한해 수집 속도 저하를 막는다.
            r    = _get_with_retry(s, url, params={"code": code, "page": page}, retries=1)
            if r is None: continue
            tbls = pd.read_html(StringIO(r.text), encoding="utf-8")
            for t in tbls:
                if "날짜" in " ".join(str(c) for c in t.columns) or \
                   "종가" in " ".join(str(c) for c in t.columns):
                    df = t.dropna(how="all")
                    all_rows.append(df)
                    break
            time.sleep(0.04)
        except Exception as e:
            logger.debug(f"OHLCV {code} p{page}: {e}")

    if not all_rows:
        return None

    df = pd.concat(all_rows, ignore_index=True).drop_duplicates()
    col_map = {}
    for c in df.columns:
        cs = str(c)
        if "날짜" in cs:    col_map[c] = "Date"
        elif "종가" in cs:  col_map[c] = "Close"
        elif "시가" in cs:  col_map[c] = "Open"
        elif "고가" in cs:  col_map[c] = "High"
        elif "저가" in cs:  col_map[c] = "Low"
        elif "거래량" in cs: col_map[c] = "Volume"
    df = df.rename(columns=col_map)

    if "Close" not in df.columns:
        return None
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    for c in ["Close","Open","High","Low","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",","").str.replace("+",""),
                errors="coerce"
            )
    df = df.dropna(subset=["Close"]).tail(PERIOD_DAYS)
    df["Amount"] = df["Close"] * df.get("Volume", pd.Series(0, index=df.index))

    try:
        with open(cache_p, "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        logger.debug(f"OHLCV 캐시 저장 실패({code}): {e}")
    return df


# ── 한국 수급 (네이버 frgn.naver) ───────────────────────────────────
#
# frgn.naver 테이블3 구조 (MultiIndex 컬럼):
#   ('날짜','날짜') ('종가','종가') ('거래량','거래량')
#   ('기관','순매매량') ('외국인','순매매량') ('외국인','보유주수') ('외국인','보유율')
#
# → 기관 순매매량 / 외국인 순매매량 컬럼을 찾아 5일/20일 합산

def _naver_supply(code: str) -> dict:
    s   = _nav()
    url = "https://finance.naver.com/item/frgn.naver"
    f5 = f20 = i5 = i20 = np.nan

    try:
        r    = _get_with_retry(s, url, params={"code": code})
        if r is None:
            return {"foreign_5d": f5, "foreign_20d": f20, "inst_5d": i5, "inst_20d": i20}
        tbls = pd.read_html(StringIO(r.text), encoding="utf-8")

        for t in tbls:
            cols_str = " ".join(str(c) for c in t.columns)
            # 외국인 또는 기관 순매매량이 있는 테이블 찾기
            if "순매매량" not in cols_str and "외국인" not in cols_str:
                continue

            df = t.copy()

            # ── MultiIndex 컬럼 처리 ──────────────────────────────
            if isinstance(df.columns, pd.MultiIndex):
                # 외국인 순매매량 컬럼
                f_col = None
                i_col = None
                for col in df.columns:
                    top, bot = str(col[0]), str(col[1])
                    if "외국인" in top and "순매매량" in bot:
                        f_col = col
                    if "기관" in top and "순매매량" in bot:
                        i_col = col
            else:
                # 일반 컬럼
                f_col = next((c for c in df.columns if "외국인" in str(c) and "순매" in str(c)), None)
                if f_col is None:
                    f_col = next((c for c in df.columns if "외국인" in str(c)), None)
                i_col = next((c for c in df.columns if "기관" in str(c) and "순매" in str(c)), None)
                if i_col is None:
                    i_col = next((c for c in df.columns if "기관" in str(c)), None)

            def _parse_col(series: pd.Series) -> pd.Series:
                return pd.to_numeric(
                    series.astype(str)
                          .str.replace(",", "")
                          .str.replace("+", "")
                          .str.strip(),
                    errors="coerce"
                ).dropna()

            if f_col is not None:
                fs = _parse_col(df[f_col])
                if len(fs) >= 5:
                    f5  = float(fs.iloc[:5].sum())
                    f20 = float(fs.iloc[:20].sum()) if len(fs) >= 20 else float(fs.sum())

            if i_col is not None:
                is_ = _parse_col(df[i_col])
                if len(is_) >= 5:
                    i5  = float(is_.iloc[:5].sum())
                    i20 = float(is_.iloc[:20].sum()) if len(is_) >= 20 else float(is_.sum())

            break   # 찾으면 종료

    except Exception as e:
        logger.debug(f"수급 {code}: {e}")

    return {"foreign_5d": f5, "foreign_20d": f20, "inst_5d": i5, "inst_20d": i20}


# ── 한국 시가총액 (네이버 종목 상세) ────────────────────────────────

def _naver_marcap_single(code: str) -> float:
    s = _nav()
    try:
        r    = _get_with_retry(s, f"https://finance.naver.com/item/main.naver?code={code}")
        if r is None: return float("nan")
        soup = BeautifulSoup(r.text, "html.parser")
        for tr in soup.select("table.no_info tr"):
            tds = tr.select("td")
            for i, td in enumerate(tds):
                if "시가총액" in td.text and i+1 < len(tds):
                    txt = tds[i+1].text.strip().replace(",","").replace("억","")
                    try:
                        return float(txt) * 1e8
                    except Exception:
                        pass
    except Exception:
        pass
    return np.nan


# ── 단일 종목 수집 ───────────────────────────────────────────────────

def _fetch_kr_one(code: str, name: str, market: str) -> Optional[dict]:
    try:
        df = _naver_ohlcv(code)
        if df is None or len(df) < 20:
            return None
        supply  = _naver_supply(code)
        mc      = _naver_marcap_single(code)
        avg_amt = float(df["Amount"].tail(20).mean()) if "Amount" in df.columns else np.nan
        return {
            "ticker":     code,
            "name":       name,
            "market":     market,
            "sector":     "",
            "ohlcv":      df,
            "market_cap": mc,
            "avg_amount": avg_amt,
            "price":      float(df["Close"].iloc[-1]),
            "is_warned":  False,
            **supply,
        }
    except Exception as e:
        logger.debug(f"KR {code}: {e}")
        return None


# ── 날짜별 수급 시계열 (백테스트용) ──────────────────────────────────
# _naver_supply는 최근 5/20일 합계만 주지만(라이브용), 백테스트에는
# '과거 각 날짜 기준' 수급이 필요하다. 아래는 frgn 페이지를 여러 장 긁어
# 날짜 인덱스를 가진 일별 외국인/기관 순매매 시계열을 만든다(캐시).
SUPPLY_PAGES = 30   # ≈ 600 거래일 (백테스트 기간 ~520일 커버)


def _supply_cols(df):
    """frgn 테이블에서 (날짜, 외국인순매매, 기관순매매) 컬럼 찾기."""
    dc = fc = ic = None
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            top, bot = str(col[0]), str(col[1])
            if "날짜" in top or "날짜" in bot: dc = col
            if "외국인" in top and "순매매" in bot: fc = col
            if "기관" in top and "순매매" in bot: ic = col
    else:
        for c in df.columns:
            cs = str(c)
            if "날짜" in cs: dc = c
            elif "외국인" in cs and "순매" in cs: fc = c
            elif "기관" in cs and "순매" in cs: ic = c
    return dc, fc, ic


def _naver_supply_series(code: str, pages: int = SUPPLY_PAGES) -> Optional[pd.DataFrame]:
    """날짜 인덱스 + [foreign, inst] 일별 순매매. 실패 시 None. 캐시됨."""
    cache_p = CACHE_DIR / f"kr_supply_{code}_{pages}p.pkl"
    if cache_p.exists():
        try:
            return pd.read_pickle(cache_p)
        except Exception:
            cache_p.unlink(missing_ok=True)

    s = _nav()
    url = "https://finance.naver.com/item/frgn.naver"
    rows = []
    for page in range(1, pages + 1):
        r = _get_with_retry(s, url, params={"code": code, "page": page}, retries=1)
        if r is None:
            break
        try:
            tbls = pd.read_html(StringIO(r.text), encoding="utf-8")
        except Exception:
            break
        got = False
        for t in tbls:
            dc, fc, ic = _supply_cols(t)
            if dc is None or (fc is None and ic is None):
                continue
            keep = [c for c in [dc, fc, ic] if c is not None]
            sub = t[keep].copy()
            sub.columns = ["date"] + (["foreign"] if fc is not None else []) + (["inst"] if ic is not None else [])
            sub["date"] = pd.to_datetime(sub["date"].astype(str).str.strip(),
                                         errors="coerce", format="%Y.%m.%d")
            sub = sub.dropna(subset=["date"])
            if len(sub):
                for c in ("foreign", "inst"):
                    if c in sub.columns:
                        sub[c] = pd.to_numeric(
                            sub[c].astype(str).str.replace(",", "").str.replace("+", "").str.strip(),
                            errors="coerce")
                rows.append(sub); got = True
                break
        if not got:
            break
    if not rows:
        return None
    out = (pd.concat(rows).drop_duplicates(subset=["date"])
           .set_index("date").sort_index())
    for c in ("foreign", "inst"):
        if c not in out.columns:
            out[c] = np.nan
    out = out[["foreign", "inst"]]
    try:
        out.to_pickle(cache_p)
    except Exception:
        pass
    return out


def attach_kr_supply_series(data: dict, pages: int = SUPPLY_PAGES, max_workers: int = 6) -> int:
    """data의 각 종목에 item['supply_series'] (날짜별 수급 DF)를 붙인다. 부착 개수 반환."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    codes = [c for c, v in data.items() if isinstance(v, dict) and "ohlcv" in v]
    n = 0
    logger.info(f"수급 시계열 수집 시작: {len(codes)}종목 × {pages}페이지 (최초 1회 느림, 이후 캐시)")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_naver_supply_series, c, pages): c for c in codes}
        for k, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                df = fut.result()
            except Exception:
                df = None
            if df is not None and len(df):
                data[c]["supply_series"] = df
                n += 1
            if k % 50 == 0:
                logger.info(f"  수급 {k}/{len(codes)}")
    logger.info(f"수급 시계열 부착 완료: {n}/{len(codes)}")
    return n


def fetch_kr_universe(
    tickers: list[str],
    progress_cb: Optional[Callable] = None,
) -> tuple[dict, list[str]]:
    """
    네이버 금융으로 한국 종목 수집.
    유니버스: KOSPI200 + KOSDAQ150 + 거래대금상위150 (약 350~450개)
    """
    # 종목명 + 시장 구분 매핑
    name_map:   dict[str, str] = {}
    market_map: dict[str, str] = {}

    kospi_codes  = {c for c, _ in _naver_marcap_top("KOSPI", 200)}
    kosdaq_codes = {c for c, _ in _naver_marcap_top("KOSDAQ", 150)}
    vol_codes    = {c: n for c, n in _naver_volume_top(150)}

    # 합집합 이름맵
    for code in kospi_codes:
        market_map[code] = "KOSPI"
    for code in kosdaq_codes:
        market_map[code] = "KOSDAQ"
    for code, name in vol_codes.items():
        market_map.setdefault(code, "KOSPI")
        name_map[code] = name

    # 이름 보충 (시총페이지에서)
    for code, name in _naver_marcap_top("KOSPI", 200) + _naver_marcap_top("KOSDAQ", 150):
        name_map[code] = name

    data:   dict = {}
    failed: list = []
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fmap = {
            ex.submit(
                _fetch_kr_one,
                t,
                name_map.get(t, t),
                market_map.get(t, "KOSPI"),
            ): t
            for t in tickers
        }
        done = 0
        for fut in as_completed(fmap):
            t = fmap[fut]
            done += 1
            if progress_cb:
                progress_cb(done, total, name_map.get(t, t))
            try:
                result = fut.result()
                if result:
                    data[t] = result
                else:
                    failed.append(t)
            except Exception as e:
                logger.debug(f"KR collect({t}): {e}")
                failed.append(t)

    logger.info(f"한국 최종: 성공 {len(data)}, 실패 {len(failed)}")
    if data:
        set_data_date(only_if_absent=True)   # 이 데이터셋의 기준일 기록(최초 1회)
    return data, failed


# ════════════════════════════════════════════════════════════════════
# 한국 벤치마크 (KOSPI / KOSDAQ 지수)
# ════════════════════════════════════════════════════════════════════

def fetch_kr_benchmarks() -> dict[str, pd.Series]:
    result = {}
    s      = _nav()
    index_map = {
        "kospi":  "https://finance.naver.com/sise/sise_index_day.naver?code=KOSPI",
        "kosdaq": "https://finance.naver.com/sise/sise_index_day.naver?code=KOSDAQ",
    }
    for name, base_url in index_map.items():
        rows = []
        for page in range(1, 28):
            try:
                r    = _nav().get(base_url + f"&page={page}", timeout=10)
                tbls = pd.read_html(StringIO(r.text), encoding="utf-8")
                for t in tbls:
                    if "날짜" in " ".join(str(c) for c in t.columns):
                        rows.append(t.dropna(how="all"))
                        break
                time.sleep(0.05)
            except Exception:
                break
        if not rows:
            continue
        df = pd.concat(rows, ignore_index=True).drop_duplicates()
        col_map = {}
        for c in df.columns:
            if "날짜" in str(c): col_map[c] = "Date"
            elif "종가" in str(c): col_map[c] = "Close"
        df = df.rename(columns=col_map)
        if "Date" not in df.columns or "Close" not in df.columns:
            continue
        df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",",""), errors="coerce")
        df = df.dropna(subset=["Date","Close"]).set_index("Date").sort_index()
        result[name] = df["Close"].tail(PERIOD_DAYS)
        logger.info(f"벤치마크 {name}: {len(result[name])}일")

    # 네이버 실패 시 yfinance fallback (KODEX200, KOSDAQ150)
    if not result:
        logger.warning("네이버 벤치마크 실패 → yfinance fallback 시도")
        try:
            import yfinance as yf
            yf_map = {"069500.KS": "kospi", "229200.KS": "kosdaq"}
            raw = yf.download(list(yf_map.keys()), period="2y",
                              auto_adjust=True, progress=False)
            for yf_code, bench_name in yf_map.items():
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        s = raw["Close"][yf_code].dropna()
                    else:
                        s = raw["Close"].dropna()
                    if len(s) > 30:
                        s.index = pd.to_datetime(s.index).tz_localize(None)
                        result[bench_name] = s
                        logger.info(f"yfinance 벤치마크 {bench_name}: {len(s)}일")
                except Exception as e2:
                    logger.debug(f"yfinance {yf_code}: {e2}")
        except Exception as e:
            logger.warning(f"yfinance fallback 실패: {e}")

    return result


# ════════════════════════════════════════════════════════════════════
# 단일 KR 종목 on-demand 수집
# ════════════════════════════════════════════════════════════════════

def _build_kr_offline_name_map() -> "dict[str, str]":
    """
    종목명 → 6자리 코드 오프라인 역매핑.
    FORCED_KR_UNIVERSE + KR_FORCED_INCLUDE 양쪽을 합산한다.
    """
    m: dict[str, str] = {}
    try:
        from data.kr_universe import FORCED_KR_UNIVERSE
        for code, name in FORCED_KR_UNIVERSE.items():
            m.setdefault(name, code)
    except Exception:
        pass
    try:
        from data.kr_universe import KR_FORCED_INCLUDE
        for code, name in KR_FORCED_INCLUDE.items():
            m.setdefault(name, code)
    except Exception:
        pass
    return m


def _naver_kr_meta(code: str) -> "tuple[str, str]":
    """
    네이버 종목 메인 페이지에서 (회사명, KOSPI|KOSDAQ) 조회.
    페이지 타이틀이 "삼성전자 코스피 | 네이버 증권" 형태임을 이용한다.
    실패 시 (code, "KOSPI") 폴백.
    """
    s = _nav()
    try:
        r = _get_with_retry(s, f"https://finance.naver.com/item/main.naver?code={code}")
        if r is None:
            return code, "KOSPI"
        soup = BeautifulSoup(r.text, "html.parser")

        title = soup.title.text if soup.title else ""
        market = "KOSDAQ" if "코스닥" in title else "KOSPI"

        # 회사명: 타이틀 정제 → h2 태그 순
        name = code
        title_clean = (
            title.replace("코스닥", "").replace("코스피", "")
                 .replace("| 네이버 증권", "").replace("| 네이버금융", "").strip()
        )
        if title_clean:
            name = title_clean
        for sel in ("div.wrap_company h2 a", "h2.h_company a", ".h_company a"):
            t = soup.select_one(sel)
            if t and t.text.strip():
                name = t.text.strip()
                break

        return name, market
    except Exception:
        return code, "KOSPI"


def fetch_single_kr_ticker_ondemand(
    query: str,
) -> "tuple[dict | None, dict[str, pd.Series], str]":
    """
    단일 KR 종목 on-demand OHLCV + 수급 + 벤치마크 수집.
    query : 6자리 종목코드 (예: '005930') 또는 FORCED_KR_UNIVERSE 내 종목명 (예: '삼성전자')
    반환  : (item, bench_close, error_msg) — 실패 시 item=None, error_msg 채움

    내부적으로 기존 _fetch_kr_one / fetch_kr_benchmarks를 재사용하며
    새 데이터 소스를 추가하지 않는다.
    """
    import re as _re

    query = str(query).strip()
    if not query:
        return None, {}, "종목코드 또는 종목명이 비어 있습니다."

    # ─ 코드 결정 ──────────────────────────────────────────────────
    if _re.match(r"^\d{6}$", query):
        code = query
        name, market = _naver_kr_meta(code)
    else:
        name_map = _build_kr_offline_name_map()
        code = name_map.get(query)
        if code is None:
            return None, {}, (
                f"'{query}': 종목명을 인식할 수 없습니다. "
                "6자리 종목코드를 직접 입력하거나 정확한 종목명(예: 삼성전자, SK하이닉스)을 사용하세요."
            )
        name = query
        _, market = _naver_kr_meta(code)

    # ─ OHLCV + 수급 + 시가총액 (기존 _fetch_kr_one 재사용) ────────
    item = _fetch_kr_one(code, name, market)
    if item is None:
        return None, {}, (
            f"{code} ({name}): 데이터 수집 실패 — "
            "상장폐지되었거나 Naver Finance에서 조회되지 않는 종목입니다."
        )

    n_days = len(item.get("ohlcv", pd.DataFrame()))
    if n_days < 60:
        return None, {}, f"{code}: OHLCV 데이터 부족 ({n_days}일, 최소 60일 필요)"

    # ─ 벤치마크 (기존 fetch_kr_benchmarks 재사용) ─────────────────
    bench_close = fetch_kr_benchmarks()
    if not bench_close:
        return None, {}, "KOSPI/KOSDAQ 벤치마크 수집 실패"

    return item, bench_close, ""
