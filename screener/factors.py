"""
screener/factors.py  v7

점수 체계 전면 개편:
  Leader Score    : 과열 포함, 가장 강한 주도주 (risk 제외)
  Buyable Score   : 지금 진입 가능한 자리
  Breakout Score  : 박스권 돌파 / 신고가 돌파 직후
  Momentum Score  : 기존 유지 (RS 기반)
  Volume Score    : 거래대금 강도 (100점)
  Quality Score   : 추세 안정성
  Risk Score      : 과열 위험 (표시용, 제외 기준 아님)
  KR Supply Score : 외국인/기관 수급
  Catalyst Proxy  : 가격/거래량 기반 촉매 신호
"""

import math
import numpy as np
import pandas as pd
from typing import Optional

try:
    from utils.scoring import calc_validated_score as _cvs, calc_validated_score_series as _cvs_series
except ImportError:
    _cvs = None
    _cvs_series = None


# ════════════════════════════════════════════════════════════════════
# 기초 지표
# ════════════════════════════════════════════════════════════════════

def _ret(close: pd.Series, d: int) -> float:
    if len(close) <= d: return float("nan")
    return (close.iloc[-1] / close.iloc[-1 - d] - 1) * 100

def _ma(close: pd.Series, w: int) -> float:
    if len(close) < w: return float("nan")
    return float(close.tail(w).mean())

def _above(close: pd.Series, w: int) -> bool:
    ma = _ma(close, w)
    return math.isfinite(ma) and bool(close.iloc[-1] > ma)

def _dist_from_ma(close: pd.Series, w: int) -> float:
    ma = _ma(close, w)
    if not math.isfinite(ma) or ma == 0: return float("nan")
    return (close.iloc[-1] / ma - 1) * 100

def _ma_slope(close: pd.Series, w: int, lookback: int = 5) -> float:
    if len(close) < w + lookback: return float("nan")
    now  = close.tail(w).mean()
    prev = close.iloc[-(w + lookback):-lookback].mean()
    if prev == 0: return float("nan")
    return (now / prev - 1) * 100

def _week52_prox(df: pd.DataFrame) -> float:
    """52주(252거래일) 고점 대비 현재가 %. 데이터 부족 시 가용 데이터 사용."""
    if "High" not in df.columns or len(df) < 5: return float("nan")
    n = min(252, len(df))  # 데이터 부족해도 가용 기간으로 계산
    h = df["High"].tail(n).max()
    if not math.isfinite(h) or h == 0: return float("nan")
    return (df["Close"].iloc[-1] / h) * 100

def _high_prox(df: pd.DataFrame, days: int) -> float:
    """N거래일 고점 대비 현재가 (%). 60d/120d/252d 각각 계산 가능."""
    if "High" not in df.columns or len(df) < 5: return float("nan")
    n = min(days, len(df))
    h = df["High"].tail(n).max()
    if not math.isfinite(h) or h == 0: return float("nan")
    return (df["Close"].iloc[-1] / h) * 100

def _multi_period_prox(df: pd.DataFrame) -> dict:
    """60d / 120d / 252d 고점 근접도 동시 계산."""
    return {
        "prox_60d":  _high_prox(df, 60),
        "prox_120d": _high_prox(df, 120),
        "prox_252d": _high_prox(df, 252),
    }

def _week52_low_prox(df: pd.DataFrame) -> float:
    """52주 저점 대비 현재가 (높을수록 저점에서 많이 반등)."""
    if "Low" not in df.columns or len(df) < 5: return float("nan")
    l52 = df["Low"].tail(252).min()
    if not math.isfinite(l52) or l52 == 0: return float("nan")
    return (df["Close"].iloc[-1] / l52 - 1) * 100

def _atr(df: pd.DataFrame, w: int = 14) -> float:
    if "High" not in df.columns or len(df) < w + 1: return float("nan")
    hi = df["High"].tail(w+1); lo = df["Low"].tail(w+1); cl = df["Close"].tail(w+1)
    tr = pd.concat([hi-lo,(hi-cl.shift(1)).abs(),(lo-cl.shift(1)).abs()],axis=1).max(axis=1).dropna()
    return float(tr.tail(w).mean())

def _ann_vol(close: pd.Series, w: int = 20) -> float:
    if len(close) < w+1: return float("nan")
    return float(close.pct_change().dropna().tail(w).std() * np.sqrt(252) * 100)

def _amount_series(df: pd.DataFrame) -> pd.Series:
    if "Amount" in df.columns: return df["Amount"]
    if "Volume" in df.columns and "Close" in df.columns:
        return df["Close"] * df["Volume"]
    return pd.Series(dtype=float)

def _dv_surge(df: pd.DataFrame) -> float:
    s = _amount_series(df).dropna()
    if len(s) < 25: return float("nan")
    return s.tail(5).mean() / s.iloc[-25:-5].mean() if s.iloc[-25:-5].mean() else float("nan")

def _dv_surge_long(df: pd.DataFrame) -> float:
    s = _amount_series(df).dropna()
    if len(s) < 80: return float("nan")
    return s.tail(20).mean() / s.iloc[-80:-20].mean() if s.iloc[-80:-20].mean() else float("nan")

def _up_dv_ratio(df: pd.DataFrame, w: int = 15) -> float:
    s = _amount_series(df)
    if len(s) < w+1 or "Close" not in df.columns: return 1.0
    sub = df.tail(w+1).copy(); sub["_amt"] = s.tail(w+1).values
    sub["_ret"] = sub["Close"].pct_change()
    up = sub[sub["_ret"] > 0]["_amt"].mean(); dn = sub[sub["_ret"] < 0]["_amt"].mean()
    if not math.isfinite(dn) or dn == 0: return 1.5 if math.isfinite(up) else 1.0
    return up / dn if math.isfinite(up) else 0.5

def _has_big_bearish(df: pd.DataFrame, thr: float = -0.05, w: int = 5) -> bool:
    if len(df) < 2: return False
    return bool((df["Close"].pct_change().tail(w) <= thr).any())

def _has_long_upper_wick(df: pd.DataFrame, w: int = 5) -> bool:
    if "High" not in df.columns or "Open" not in df.columns: return False
    for _, row in df.tail(w).iterrows():
        body = abs(row["Close"] - row.get("Open", row["Close"]))
        wick = row["High"] - max(row["Close"], row.get("Open", row["Close"]))
        if body > 0 and wick > body * 2: return True
    return False

def _vol_up_ret_weak(df: pd.DataFrame, thr: float = 3.0) -> bool:
    s = _amount_series(df)
    if len(s) < 21: return False
    avg = s.iloc[-21:-1].mean()
    if not math.isfinite(avg) or avg == 0: return False
    return (s.iloc[-1] / avg >= thr) and (df["Close"].pct_change().iloc[-1] < 0)

def _consecutive_up(close: pd.Series, w: int = 7) -> int:
    cnt = 0
    for r in reversed(close.pct_change().tail(w).values):
        if r > 0: cnt += 1
        else: break
    return cnt

def _recent_pullback(close: pd.Series, df: pd.DataFrame, days: int = 7) -> bool:
    s = _amount_series(df)
    if len(close) < days+5 or len(s) < days+5: return False
    price_down = close.pct_change().tail(days).mean() < 0
    avg_before = s.iloc[-(days+5):-days].mean(); avg_recent = s.tail(days).mean()
    if not math.isfinite(avg_before) or avg_before == 0: return False
    return bool(price_down and avg_recent < avg_before)

def _rs(a: float, b: float) -> float:
    return float("nan") if not (math.isfinite(a) and math.isfinite(b)) else a - b

def _clamp(v, lo, hi, neutral=0):
    if not math.isfinite(float(v) if v is not None else float("nan")): return neutral
    return max(lo, min(hi, float(v)))


# ════════════════════════════════════════════════════════════════════
# 1. Leader Score  (0-100) — risk 제외, 주도주 강도만
# ════════════════════════════════════════════════════════════════════

def leader_score(
    close: pd.Series,
    df: pd.DataFrame,
    rs_market_20: float,
    rs_market_60: float,
    rs_sector_20: float,
    prox52: float,
    rs_rank_pct: float = 0.5,
    market_cap: float = float("nan"),
) -> float:
    """
    가장 강한 주도주 포착용.
    risk를 빼지 않음 — MU/SNDK 같은 과열 주도주도 높은 점수.
    """
    r20 = _ret(close, 20); r60 = _ret(close, 60)
    surge = _dv_surge(df)

    # 수익률 강도 (0~35점)
    s_r20 = _clamp(r20, -10, 30, 0) / 30 * 20
    s_r60 = _clamp(r60, -20, 60, 0) / 60 * 15

    # 시장 대비 RS (0~25점) — percentile 중심
    s_rs_abs = _clamp(rs_market_20, -15, 25, 0) / 25 * 12
    s_rs_pct = rs_rank_pct * 13  # percentile이 핵심

    # 섹터 대비 RS (0~10점)
    s_rs_sect = _clamp(rs_sector_20, -10, 20, 0) / 20 * 10

    # 52주 고점 근접 (0~10점) — 높을수록 주도주
    s_prox = (_clamp(prox52, 60, 100, 75) - 60) / 40 * 10

    # 거래대금 강도 (0~10점)
    s_vol = _clamp((surge - 1) / 2, 0, 1, 0) * 10 if math.isfinite(surge) else 5

    # MA 위치 (0~5점)
    ma_pts = (2 if _above(close, 20) else 0) + (2 if _above(close, 50) else 0) + (1 if _above(close, 200) else 0)

    return round(max(0, min(100, s_r20+s_r60+s_rs_abs+s_rs_pct+s_rs_sect+s_prox+s_vol+ma_pts)), 1)


# ════════════════════════════════════════════════════════════════════
# 2. Momentum Score  (0-100) — 기존 유지
# ════════════════════════════════════════════════════════════════════

def momentum_score(
    close: pd.Series,
    rs_market_20: float, rs_market_60: float,
    rs_sector_20: float, prox52: float,
    rs_rank_pct: float = 0.5,
) -> float:
    r20 = _ret(close, 20); r60 = _ret(close, 60)
    s_r20 = _clamp(r20, -15, 25, 0) / 25 * 28
    s_r60 = _clamp(r60, -20, 40, 0) / 40 * 18
    s_rs_abs = (_clamp(rs_market_20,-15,20,0)+_clamp(rs_market_60,-20,30,0)/1.5)/2/20*12
    s_rs_pct = rs_rank_pct * 10
    s_rs_sect = _clamp(rs_sector_20,-10,15,0)/15*10
    s_prox = (_clamp(prox52,60,100,75)-60)/40*10
    ma_pts = (2 if _above(close,20) else 0)+(2 if _above(close,50) else 0)+(1 if _above(close,200) else 0)
    return round(max(0,min(100,s_r20+s_r60+s_rs_abs+s_rs_pct+s_rs_sect+s_prox+ma_pts)),1)


# ════════════════════════════════════════════════════════════════════
# 3. Volume Score  (0-100)
# ════════════════════════════════════════════════════════════════════

def volume_score(df: pd.DataFrame, market_cap: float = float("nan")) -> float:
    surge5  = _dv_surge(df); surge20 = _dv_surge_long(df); up_ratio = _up_dv_ratio(df)
    s_surge5  = _clamp((surge5-0.5)/1.5*35,0,35,17.5)  if math.isfinite(surge5)  else 17.5
    s_surge20 = _clamp((surge20-0.8)/0.7*20,0,20,10)   if math.isfinite(surge20) else 10
    s_upvol   = _clamp((up_ratio-0.5)/1.5*30,0,30,15)
    s_turn    = 5.0
    s = _amount_series(df)
    if math.isfinite(market_cap) and market_cap > 0 and not s.empty:
        avg_dv = s.tail(5).mean()
        if math.isfinite(avg_dv) and avg_dv > 0:
            s_turn = _clamp(avg_dv/market_cap*252*10,0,10,5)
    penalty = 15 if _vol_up_ret_weak(df, thr=3.0) else 0
    return round(max(0,min(100,s_surge5+s_surge20+s_upvol+s_turn-penalty)),1)


# ════════════════════════════════════════════════════════════════════
# 4. Buyable Score  (0-100) — 진입 자리
# ════════════════════════════════════════════════════════════════════

def buyable_score(close: pd.Series, df: pd.DataFrame, prox52: float) -> float:
    dist20 = _dist_from_ma(close, 20); dist50 = _dist_from_ma(close, 50)
    r5 = _ret(close, 5); consec = _consecutive_up(close)
    score = 50.0
    if math.isfinite(prox52):
        if prox52 >= 95: score += 15
        elif prox52 >= 90: score += 10
        elif prox52 >= 80: score += 5
    if math.isfinite(dist20):
        if 0 <= dist20 <= 10: score += 15
        elif 10 < dist20 <= 15: score += 8
        elif -5 <= dist20 < 0: score += 5
    if _recent_pullback(close, df, days=5): score += 10
    if len(close) >= 3:
        lr = close.pct_change().iloc[-1]; pr = close.pct_change().iloc[-2]
        if lr > 0 and pr > 0: score += 5
        elif lr > 0.01: score += 3
    if consec <= 2: score += 5
    if math.isfinite(dist20):
        if dist20 >= 20: score -= 25
        elif dist20 >= 15: score -= 15
    if math.isfinite(dist50):
        if dist50 >= 30: score -= 15
        elif dist50 >= 20: score -= 8
    if math.isfinite(r5):
        if r5 >= 15: score -= 20
        elif r5 >= 10: score -= 10
    if _has_big_bearish(df, thr=-0.05, w=3): score -= 15
    if _has_long_upper_wick(df, w=3): score -= 10
    if consec >= 4: score -= 15
    elif consec == 3: score -= 8
    return round(max(0,min(100,score)),1)


# ════════════════════════════════════════════════════════════════════
# 5. Breakout Score  (0-100) — 박스권/신고가 돌파
# ════════════════════════════════════════════════════════════════════

def breakout_score(close: pd.Series, df: pd.DataFrame) -> float:
    """
    박스권 돌파 / 52주 신고가 돌파 / 거래량 폭증 패턴.
    MU/SNDK 같은 조용한 매집 후 폭발 포착.
    """
    score = 0.0
    if len(close) < 60: return score

    # 52주 신고가 돌파 여부 (가장 중요)
    high_20d = close.tail(20).max()
    high_60d = close.tail(60).max()
    cur = close.iloc[-1]

    if "High" in df.columns:
        high_52w = df["High"].tail(252).max()
        # 현재가가 52주 고점의 95% 이상
        if cur >= high_52w * 0.95:
            score += 30
        elif cur >= high_52w * 0.90:
            score += 15

    # 20일 박스권 상단 돌파 (최근 3일 안에 돌파)
    box_top = close.iloc[-23:-3].max() if len(close) > 25 else close.iloc[0]
    if cur > box_top * 1.02:
        score += 20
    elif cur > box_top:
        score += 10

    # 돌파 시 거래량 폭증 (돌파의 신뢰도)
    surge = _dv_surge(df)
    if math.isfinite(surge):
        if surge >= 3.0: score += 25
        elif surge >= 2.0: score += 15
        elif surge >= 1.5: score += 8

    # 변동성 축소 후 폭발 패턴
    # (직전 10일 변동성이 낮았다가 최근 3일 크게 오름)
    if len(close) >= 30:
        vol_before = close.iloc[-30:-10].pct_change().std()
        vol_recent = close.tail(5).pct_change().std()
        if math.isfinite(vol_before) and math.isfinite(vol_recent) and vol_before > 0:
            vol_ratio = vol_recent / vol_before
            if vol_ratio >= 2.5: score += 15
            elif vol_ratio >= 1.5: score += 8

    # 매집 패턴: 횡보 + 거래량 누적 증가
    # (20일 수익률은 낮지만 거래대금이 꾸준히 증가)
    r20 = _ret(close, 20)
    surge_long = _dv_surge_long(df)
    if math.isfinite(r20) and math.isfinite(surge_long):
        if abs(r20) < 5 and surge_long >= 1.3:
            score += 10  # 횡보+매집 신호

    return round(max(0, min(100, score)), 1)


# ════════════════════════════════════════════════════════════════════
# 6. Top Risk Score  (0-100, 높을수록 위험) — 표시용만, 제외 기준 아님
# ════════════════════════════════════════════════════════════════════

def top_risk_score(close: pd.Series, df: pd.DataFrame) -> float:
    dist20 = _dist_from_ma(close, 20); dist50 = _dist_from_ma(close, 50)
    r5 = _ret(close, 5); consec = _consecutive_up(close); surge = _dv_surge(df)
    score = 0.0
    if math.isfinite(dist20):
        if dist20 >= 25: score += 30
        elif dist20 >= 20: score += 20
        elif dist20 >= 15: score += 10
    if math.isfinite(dist50):
        if dist50 >= 35: score += 20
        elif dist50 >= 25: score += 12
        elif dist50 >= 18: score += 6
    if consec >= 5: score += 20
    elif consec >= 4: score += 12
    elif consec == 3: score += 6
    if math.isfinite(r5):
        if r5 >= 15: score += 25
        elif r5 >= 10: score += 15
        elif r5 >= 7: score += 8
    if _vol_up_ret_weak(df, thr=3.0): score += 20
    elif _vol_up_ret_weak(df, thr=2.0): score += 12
    if _has_big_bearish(df, thr=-0.06, w=3): score += 15
    if _has_long_upper_wick(df, w=3): score += 10
    if math.isfinite(r5) and r5 >= 5 and math.isfinite(surge) and surge < 0.8:
        score += 10
    return round(max(0,min(100,score)),1)


# ════════════════════════════════════════════════════════════════════
# 7. Quality of Trend Score  (0-100)
# ════════════════════════════════════════════════════════════════════

def quality_of_trend_score(close: pd.Series, df: pd.DataFrame) -> float:
    r20=_ret(close,20); r60=_ret(close,60)
    vol20=_ann_vol(close,20); vol60=_ann_vol(close,60)
    score=50.0
    if math.isfinite(r20) and math.isfinite(vol20) and vol20>0:
        score += _clamp(r20/(vol20/np.sqrt(252/20))*4,-10,15,0)
    if math.isfinite(r60) and math.isfinite(vol60) and vol60>0:
        score += _clamp(r60/(vol60/np.sqrt(252/60))*3,-8,12,0)
    if len(close)>=60:
        c60=close.tail(60); mdd=float(((c60/c60.cummax())-1).min()*100)
        if mdd<-20: score-=20
        elif mdd<-15: score-=12
        elif mdd<-10: score-=6
        elif mdd>-5: score+=8
    sl20=_ma_slope(close,20,5); sl50=_ma_slope(close,50,10)
    if math.isfinite(sl20):
        score += 8 if sl20>0 else -5
    if math.isfinite(sl50):
        score += 7 if sl50>0 else -5
    if len(close)>=20:
        ma20s=close.rolling(20).mean()
        ratio=int((close.tail(20)>ma20s.tail(20)).sum())/20
        score += (ratio-0.5)*20
    return round(max(0,min(100,score)),1)


# ════════════════════════════════════════════════════════════════════
# 8. Catalyst Proxy Score  (0-100) — 가격/거래량만으로 촉매 신호 추정
# ════════════════════════════════════════════════════════════════════

def catalyst_proxy_score(close: pd.Series, df: pd.DataFrame) -> float:
    """
    외부 뉴스/실적 API 없이 가격+거래량 패턴으로 촉매 신호 추정.
    실적 발표 후 갭상승, 신고가+거래량 폭증, 섹터 동반 상승 패턴 등.
    """
    score = 50.0
    if len(close) < 10: return score

    r1  = _ret(close, 1)   # 1일
    r3  = _ret(close, 3)   # 3일
    r5  = _ret(close, 5)   # 5일
    surge = _dv_surge(df)

    # 갭상승 후 고가 마감 (실적 서프라이즈 패턴)
    if "Open" in df.columns and len(df) >= 2:
        gap = (df["Close"].iloc[-1] / df["Open"].iloc[-1] - 1) * 100
        prev_close = df["Close"].iloc[-2]
        gap_up = (df["Open"].iloc[-1] / prev_close - 1) * 100
        if gap_up >= 3 and gap >= 0:   # 갭상승 후 양봉 마감
            score += 20
        elif gap_up >= 5:              # 큰 갭상승
            score += 15

    # 신고가 + 거래량 폭증 (수급 몰림)
    prox52 = _week52_prox(df)
    if math.isfinite(prox52) and prox52 >= 98 and math.isfinite(surge) and surge >= 2:
        score += 20
    elif math.isfinite(prox52) and prox52 >= 95 and math.isfinite(surge) and surge >= 1.5:
        score += 12

    # 연속 강세 (3~5일) + 거래량 지속
    r3_pos = math.isfinite(r3) and r3 >= 8
    surge_pos = math.isfinite(surge) and surge >= 1.3
    if r3_pos and surge_pos:
        score += 15

    # 52주 저점 대비 크게 반등 (턴어라운드 신호)
    low_prox = _week52_low_prox(df)
    if math.isfinite(low_prox):
        if 15 <= low_prox <= 40:   # 저점 대비 15~40% 반등 (초기)
            score += 10
        elif low_prox < 15:        # 아직 저점 근처
            score -= 10

    # 돌파 후 음봉 없이 버팀 (건강한 상승)
    if math.isfinite(r5) and r5 >= 5:
        recent_returns = close.pct_change().tail(5)
        neg_days = (recent_returns < -0.02).sum()
        if neg_days == 0:
            score += 10
        elif neg_days >= 3:
            score -= 10

    return round(max(0, min(100, score)), 1)


# ════════════════════════════════════════════════════════════════════
# 9. KR Supply Score  (0-100)
# ════════════════════════════════════════════════════════════════════

def kr_supply_score(
    foreign_5d: float, foreign_20d: float,
    inst_5d: float, inst_20d: float,
    market_cap: float, avg_amount: float,
) -> float:
    score = 50.0

    def _sign_score(v5, v20, base_pos, base_neg):
        s = 0
        if math.isfinite(float(v5 if v5 else float("nan"))):
            v5 = float(v5)
            if v5 > 0: s += base_pos
            elif v5 < 0: s += base_neg
        if math.isfinite(float(v20 if v20 else float("nan"))):
            v20 = float(v20)
            if v20 > 0: s += base_pos * 0.5
            elif v20 < 0: s += base_neg * 0.5
        return s

    score += _sign_score(foreign_5d, foreign_20d, +15, -12)
    score += _sign_score(inst_5d,    inst_20d,    +12, -10)
    f_pos = math.isfinite(float(foreign_5d if foreign_5d else float("nan"))) and float(foreign_5d) > 0
    i_pos = math.isfinite(float(inst_5d    if inst_5d    else float("nan"))) and float(inst_5d)    > 0
    if f_pos and i_pos: score += 10
    elif not f_pos and not i_pos and \
         math.isfinite(float(foreign_5d if foreign_5d else float("nan"))) and \
         math.isfinite(float(inst_5d    if inst_5d    else float("nan"))):
        score -= 8
    return round(max(0, min(100, score)), 1)


# ════════════════════════════════════════════════════════════════════
# Market Regime
# ════════════════════════════════════════════════════════════════════

def market_regime_score(bench_close: dict) -> dict:
    def _ab(key, w):
        s = bench_close.get(key)
        if s is None or len(s) < w: return None
        return bool(s.iloc[-1] > s.tail(w).mean())

    us_score = 50
    for key in ["SPY","QQQ"]:
        for w in [20,50,200]:
            ab = _ab(key, w)
            if ab is True: us_score += 4
            elif ab is False: us_score -= 4
    sector_etfs = ["XLK","XLI","XLF","XLE","XLV","XLC","XLY","XLU","XLB","XLRE"]
    above_cnt = sum(1 for e in sector_etfs if _ab(e,20) is True)
    us_score += (above_cnt/len(sector_etfs)-0.5)*20
    us_score = max(0, min(100, us_score))

    kr_score = 50
    for key in ["kospi","kosdaq"]:
        for w in [20,60,200]:
            ab = _ab(key, w)
            if ab is True: kr_score += 6
            elif ab is False: kr_score -= 6
    kr_score = max(0, min(100, kr_score))

    return {
        "us_regime":  round(us_score,1), "kr_regime":  round(kr_score,1),
        "us_haircut": round(max(0,(50-us_score)/50*0.1),3),
        "kr_haircut": round(max(0,(50-kr_score)/50*0.1),3),
    }


# ════════════════════════════════════════════════════════════════════
# 트랙 분류
# ════════════════════════════════════════════════════════════════════

def classify_track(
    leader: float, buyable: float, breakout: float,
    risk: float, momentum: float,
) -> str:
    """
    3개 트랙 분류:
    - Hot Leader:    강력한 주도주 (과열 포함)
    - Leader Watch:  주도주지만 아직 확인 필요
    - Buyable:       진입 자리 좋은 종목
    - Breakout:      막 돌파하는 종목
    - Pullback:      눌림목 대기
    - Watch Only:    관심만
    - Avoid:         제외
    """
    # Hot Leader — 가장 강한 주도주 (과열이어도 포함)
    if leader >= 75 and momentum >= 60:
        if risk >= 65:
            return "Hot Leader / Extended"
        else:
            return "Hot Leader / Buyable"

    # 신호 돌파 — 막 터지는 종목
    if breakout >= 65 and momentum >= 45:
        return "Breakout Signal"

    # 좋은 종목 + 진입 자리
    if leader >= 60 and buyable >= 60 and risk < 55:
        return "Leader / Buyable"

    # 좋은 종목 + 눌림목 대기
    if leader >= 55 and buyable < 50 and risk < 60:
        return "Leader / Pullback Wait"

    # 모멘텀 약
    if momentum < 30:
        return "Avoid / Weak"

    # 관심
    if leader >= 50:
        return "Watch Only"

    return "Avoid / Weak"


# ════════════════════════════════════════════════════════════════════
# 강점/약점 요약
# ════════════════════════════════════════════════════════════════════

def build_summary(row: dict, market: str = "US") -> str:
    leader   = row.get("leader_score",   0)
    mom      = row.get("momentum_score", 0)
    vol      = row.get("volume_score",   0)
    buy      = row.get("buyable_score",  0)
    risk     = row.get("top_risk_score", 0)
    breakout = row.get("breakout_score", 0)
    catalyst = row.get("catalyst_score", 0)
    track    = row.get("track",          "")

    strengths = []
    weaknesses = []

    if leader >= 75:  strengths.append("강력한 주도주")
    elif leader >= 60: strengths.append("주도주 후보")

    if breakout >= 65: strengths.append("돌파 신호")
    elif breakout >= 45: strengths.append("돌파 준비 중")

    if vol >= 70:  strengths.append("거래대금 급증")
    elif vol >= 55: strengths.append("거래대금 증가")

    if catalyst >= 70: strengths.append("촉매 신호 감지")

    r20 = row.get("ret_20d", float("nan"))
    prox = row.get("week52_prox", float("nan"))
    if math.isfinite(prox) and prox >= 95:  strengths.append(f"52주 신고가 {prox:.0f}%")
    if math.isfinite(r20)  and r20 >= 10:   strengths.append(f"20일 +{r20:.1f}%")
    if buy >= 65: strengths.append("진입 자리 좋음")

    if market == "KR":
        f5 = row.get("foreign_5d", float("nan")); i5 = row.get("inst_5d", float("nan"))
        if math.isfinite(f5) and f5 > 0 and math.isfinite(i5) and i5 > 0:
            strengths.append("외국인+기관 동시 순매수")
        elif math.isfinite(f5) and f5 > 0: strengths.append("외국인 순매수")

    if risk >= 70:  weaknesses.append(f"과열 주의 (Risk {risk:.0f})")
    elif risk >= 50: weaknesses.append("추격 위험")
    if buy <= 30:   weaknesses.append("지금 진입 자리 나쁨")

    dist20 = row.get("dist_20dma", float("nan"))
    if math.isfinite(dist20) and dist20 >= 20: weaknesses.append(f"MA20 이격 {dist20:.0f}%")
    if mom < 35:    weaknesses.append("상대강도 약함")

    s_text = " · ".join(strengths[:3]) if strengths else "특이사항 없음"
    w_text = " · ".join(weaknesses[:2]) if weaknesses else "없음"
    return f"강점: {s_text}  /  주의: {w_text}"


# ════════════════════════════════════════════════════════════════════
# US/KR 공통 헬퍼 (calc_us_factors / calc_kr_factors 가 공유)
#   — 두 함수에 그대로 중복돼 있던 동일 로직만 추출. 시장별로 다른 부분
#     (벤치마크 키, MA 윈도우, 수급/신규팩터)은 각 함수에 그대로 남겨둔다.
# ════════════════════════════════════════════════════════════════════

def _bench_ret(bench_close: dict, key: str, d: int) -> float:
    """벤치마크 key의 d거래일 수익률(%). 없거나 이력 부족 시 NaN."""
    s = bench_close.get(key)
    if s is None or len(s) <= d:
        return float("nan")
    return (s.iloc[-1] / s.iloc[-1-d] - 1) * 100


def _sector_strength(sector_avg_ret20: float) -> float:
    """섹터 평균 20일 수익률 → 0~100 강도. (US/KR 동일 공식)"""
    if math.isfinite(sector_avg_ret20):
        return max(0, min(100, 50 + sector_avg_ret20 * 2))
    return 50


def _core_factor_scores(
    close, df, rs_20: float, rs_60: float, rs_sect: float,
    prox52: float, rs_rank_pct: float, mc: float,
) -> dict:
    """
    US/KR 공통의 8개 코어 팩터 점수.
    rs_20/rs_60/rs_sect 는 각 시장의 벤치마크 기준으로 미리 계산해 넘긴다
    (US=SPY/QQQ/섹터평균, KR=KOSPI·KOSDAQ/섹터평균). 호출 형태는 동일.
    """
    return {
        "leader":   leader_score(close, df, rs_20, rs_60, rs_sect, prox52, rs_rank_pct, mc),
        "momentum": momentum_score(close, rs_20, rs_60, rs_sect, prox52, rs_rank_pct),
        "volume":   volume_score(df, mc),
        "buyable":  buyable_score(close, df, prox52),
        "breakout": breakout_score(close, df),
        "risk":     top_risk_score(close, df),
        "quality":  quality_of_trend_score(close, df),
        "catalyst": catalyst_proxy_score(close, df),
    }


# ════════════════════════════════════════════════════════════════════
# 미국 팩터 계산
# ════════════════════════════════════════════════════════════════════

def calc_us_factors(
    ticker: str, item: dict, bench_close: dict,
    sector_bench: str, sector_avg_ret20: float, sector_avg_ret60: float,
    rs_rank_pct: float = 0.5, regime: dict = None,
) -> dict:
    df    = item["ohlcv"]
    close = df["Close"]
    if "Amount" not in df.columns:
        df = df.copy(); df["Amount"] = df["Close"] * df["Volume"]

    def bret(key, d):
        return _bench_ret(bench_close, key, d)

    r5  = _ret(close,5); r20 = _ret(close,20); r60 = _ret(close,60)
    rs_spy_20  = _rs(r20, bret("SPY",20))
    rs_spy_60  = _rs(r60, bret("SPY",60))
    rs_qqq_20  = _rs(r20, bret("QQQ",20))
    rs_sect_20 = _rs(r20, bret(sector_bench,20)) if sector_bench else float("nan")
    rs_sect_avg = _rs(r20, sector_avg_ret20)

    prox52   = _week52_prox(df)
    dist20   = _dist_from_ma(close,20); dist50 = _dist_from_ma(close,50)
    above20  = _above(close,20); above50 = _above(close,50); above200 = _above(close,200)
    surge    = _dv_surge(df); ann_vol = _ann_vol(close)
    mc       = item.get("market_cap", float("nan"))

    # 다중 기간 고점 근접도 (60d/120d/252d) — 데이터 길이에 따라 자동 선택
    multi_prox = _multi_period_prox(df)
    prox_60d  = multi_prox["prox_60d"]
    prox_120d = multi_prox["prox_120d"]
    prox_252d = multi_prox["prox_252d"]
    # 가용 데이터 길이에 따른 best prox (252일 미만이면 가장 긴 기간 사용)
    best_prox = prox_252d if math.isfinite(prox_252d) else prox_120d if math.isfinite(prox_120d) else prox_60d

    # 7개 점수 계산 — US/KR 공통 코어 (공유 헬퍼)
    _sc = _core_factor_scores(close, df, rs_spy_20, rs_spy_60, rs_sect_avg, prox52, rs_rank_pct, mc)
    ldr  = _sc["leader"];   mom  = _sc["momentum"]; vol = _sc["volume"]
    buy  = _sc["buyable"];  brk  = _sc["breakout"]; risk = _sc["risk"]
    qot  = _sc["quality"];  cat  = _sc["catalyst"]

    # 섹터 약한데 혼자 급등 → risk 가산
    if math.isfinite(rs_sect_avg) and rs_sect_avg > 10 and \
       math.isfinite(bret(sector_bench,20)) and bret(sector_bench,20) < 0:
        risk = min(100, risk + 8)

    sect_str = _sector_strength(sector_avg_ret20)
    haircut  = (regime or {}).get("us_haircut", 0)

    # ── 최적 가중치 로드 + breakout 포함한 공식으로 통일 ──────────
    w = None
    try:
        from screener.backtest import load_optimal_weights
        w = load_optimal_weights()
        w_lead = w.get("leader",   0.38)
        w_vol  = w.get("volume",   0.18)
        w_brk  = w.get("breakout", 0.12)   # breakout 포함 (기존 누락)
        w_cat  = w.get("catalyst", 0.15)
        w_qot  = w.get("quality",  0.10)
        w_sect = max(0.03, 1.0 - w_lead - w_vol - w_brk - w_cat - w_qot)
        w_buy  = w.get("buyable",  0.30)   # buyable_final에서 더 강하게
        w_mom  = w.get("momentum", 0.25)
        w_risk = w.get("risk_penalty", 0.10)
    except Exception:
        w = {"leader":0.38,"volume":0.18,"breakout":0.12,"catalyst":0.15,"quality":0.10,"risk_penalty":0.10}
        w_lead=0.38; w_vol=0.18; w_brk=0.12; w_cat=0.15; w_qot=0.10; w_sect=0.07
        w_buy=0.30;  w_mom=0.25; w_risk=0.10

    # quality 팩터 제거 — 한국·미국 백테스트에서 수익률·급등 적중 양쪽 모두
    # 역효과(-)로 판정됨. 캐시된 가중치까지 덮어 강제 0으로 고정한다.
    # (leader_final·buyable_final·validated_score 세 경로 모두 반영)
    w = {**w, "quality": 0.0}
    w_qot = 0.0

    # Leader Final — breakout_score 포함 (백테스트 최적화 공식과 일치)
    leader_final = round(max(0, min(100,
        ldr  * w_lead +
        vol  * w_vol  +
        brk  * w_brk  +   # breakout 반영
        cat  * w_cat  +
        qot  * w_qot  +
        sect_str * w_sect
    ) * (1-haircut)), 1)

    # Buyable Final — buyable_score를 강하게 반영 (목적에 맞게 분리)
    buyable_final = round(max(0, min(100,
        mom  * w_mom  +
        buy  * w_buy  +   # buyable 강화
        vol  * w_vol  +
        qot  * w_qot  +
        cat  * 0.10   +
        sect_str * 0.05 -
        risk * w_risk
    ) * (1-haircut)), 1)

    track = classify_track(ldr, buy, brk, risk, mom)

    risk_flag = ""
    if risk >= 65:                         risk_flag = "추격위험"
    elif _has_big_bearish(df, w=3):        risk_flag = "장대음봉"
    elif _has_long_upper_wick(df, w=3):    risk_flag = "긴윗꼬리"

    row = {
        "ticker":          ticker,
        "company":         item.get("company", ticker),
        "sector":          item.get("sector",""),
        "industry":        item.get("industry",""),
        "price":           round(item.get("price", float("nan")),2),
        "market_cap":      mc,
        "avg_dv":          item.get("avg_dv", float("nan")),
        "earnings_date":   item.get("earnings_date",""),
        "ret_5d":          round(r5,2),   "ret_20d": round(r20,2), "ret_60d": round(r60,2),
        "rs_spy_20":       round(rs_spy_20,2),
        "rs_qqq_20":       round(rs_qqq_20,2),
        "rs_sector_20":    round(rs_sect_20,2),
        "rs_sector_vs_avg":round(rs_sect_avg,2),
        "rs_rank_pct":     round(rs_rank_pct*100,1),
        "vol_surge":       round(surge,3) if math.isfinite(surge) else float("nan"),
        "week52_prox":     round(prox52,2) if math.isfinite(prox52) else float("nan"),
        "week52_low_prox": round(_week52_low_prox(df),2),
        "prox_60d":        round(prox_60d,2)  if math.isfinite(prox_60d)  else float("nan"),
        "prox_120d":       round(prox_120d,2) if math.isfinite(prox_120d) else float("nan"),
        "prox_252d":       round(prox_252d,2) if math.isfinite(prox_252d) else float("nan"),
        "above_20dma":     above20, "above_50dma": above50, "above_200dma": above200,
        "dist_20dma":      round(dist20,2) if math.isfinite(dist20) else float("nan"),
        "dist_50dma":      round(dist50,2) if math.isfinite(dist50) else float("nan"),
        "ann_vol":         round(ann_vol,2) if math.isfinite(ann_vol) else float("nan"),
        "leader_score":    ldr,
        "momentum_score":  mom,
        "volume_score":    vol,
        "buyable_score":   buy,
        "breakout_score":  brk,
        "top_risk_score":  risk,
        "quality_score":   qot,
        "catalyst_score":  cat,
        "leader_final":    leader_final,
        "buyable_final":   buyable_final,
        "final_score":     leader_final,   # 기본 정렬은 leader_final
        "track":           track,
        "risk_flag":       risk_flag,
        "yf_link":         f"https://finance.yahoo.com/quote/{ticker}",
        "news_link":       f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws",
        "edgar_link":      f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K",
    }
    row["summary"] = build_summary(row, "US")

    # validated_score: 백테스트와 동일한 공식 (supply/sector/재무 보정 제외)
    if _cvs_series is not None:
        _score_row = {
            "leader":   row.get("leader_score",   50),
            "volume":   row.get("volume_score",   50),
            "breakout": row.get("breakout_score", 50),
            "catalyst": row.get("catalyst_score", 50),
            "quality":  row.get("quality_score",  50),
            "risk":     row.get("top_risk_score", 30),
            "ret_20d":  row.get("ret_20d",         0),
        }
        row["validated_score"] = float(_cvs_series(pd.DataFrame([_score_row]), weights=w).iloc[0])
    elif _cvs is not None:
        row["validated_score"] = _cvs(
            leader   = float(row.get("leader_score",   50) or 50),
            volume   = float(row.get("volume_score",   50) or 50),
            breakout = float(row.get("breakout_score", 50) or 50),
            catalyst = float(row.get("catalyst_score", 50) or 50),
            quality  = float(row.get("quality_score",  50) or 50),
            risk     = float(row.get("top_risk_score", 30) or 30),
            ret_20d  = float(row.get("ret_20d",         0) or 0),
            weights  = w,
        )
    else:
        row["validated_score"] = row.get("leader_score", 50)

    return row


# ════════════════════════════════════════════════════════════════════
# 한국 팩터 계산
# ════════════════════════════════════════════════════════════════════

def calc_kr_factors(
    ticker: str, item: dict, bench_close: dict,
    sector_avg_ret20: float, rs_rank_pct: float = 0.5, regime: dict = None,
) -> dict:
    df    = item["ohlcv"]; close = df["Close"]

    def bret(key, d):
        return _bench_ret(bench_close, key, d)

    r5=_ret(close,5); r20=_ret(close,20); r60=_ret(close,60)
    mkt = item.get("market","KOSPI"); bk = "kospi" if mkt=="KOSPI" else "kosdaq"
    rs_mkt_20 = _rs(r20, bret(bk,20)); rs_mkt_60 = _rs(r60, bret(bk,60))
    rs_sect_20 = _rs(r20, sector_avg_ret20)

    prox52=_week52_prox(df); dist20=_dist_from_ma(close,20); dist60=_dist_from_ma(close,60)
    above20=_above(close,20); above60=_above(close,60); above200=_above(close,200)
    surge=_dv_surge(df); ann_vol=_ann_vol(close)
    mc=item.get("market_cap",float("nan")); avg_amt=item.get("avg_amount",float("nan"))

    # 8개 코어 점수 — US/KR 공통 (공유 헬퍼)
    _sc = _core_factor_scores(close, df, rs_mkt_20, rs_mkt_60, rs_sect_20, prox52, rs_rank_pct, mc)
    ldr  = _sc["leader"];   mom  = _sc["momentum"]; vol = _sc["volume"]
    buy  = _sc["buyable"];  brk  = _sc["breakout"]; risk = _sc["risk"]
    qot  = _sc["quality"];  cat  = _sc["catalyst"]
    sup  = kr_supply_score(
        item.get("foreign_5d",float("nan")), item.get("foreign_20d",float("nan")),
        item.get("inst_5d",float("nan")),    item.get("inst_20d",float("nan")),
        mc, avg_amt,
    )

    # ── 신규 factor (급등주 탐지 강화) ──────────────────────────────
    mom_accel  = momentum_accel_score(close)
    vol_shock  = volume_shock_score(df)
    brk_persist= breakout_persistence_score(close, df)
    near_hi    = near_high_score(df)
    near_hi_comp = near_hi["composite"]
    vol_contract= volatility_contraction_score(close, df)
    pullback_q  = pullback_quality_score(close, df)
    risk_filter = risk_filter_score(close, df)

    # 섹터 상대강도 (독립 factor)
    sect_rs = sector_relative_strength_score(
        rs_sector_20    = rs_sect_20,
        rs_rank_pct     = rs_rank_pct,
        sector_avg_ret20= sector_avg_ret20,
    )

    # 수급 흐름 연속성 (독립 factor)
    sup_flow = supply_flow_score(
        foreign_5d  = item.get("foreign_5d",  float("nan")),
        foreign_20d = item.get("foreign_20d", float("nan")),
        inst_5d     = item.get("inst_5d",     float("nan")),
        inst_20d    = item.get("inst_20d",    float("nan")),
        foreign_consecutive = item.get("foreign_consecutive", 0),
        inst_consecutive    = item.get("inst_consecutive",    0),
    )   # 재정의된 risk (저유동성/급락 필터)

    # ── 재정의 volume: 거래대금 급증 + 가격 상승 동반 강화 ─────────
    vol_enhanced = round(
        vol * 0.4 + vol_shock * 0.6, 1  # 기존 volume 40% + 새 volume_shock 60%
    )

    # ── 재정의 breakout: 돌파 + 지속력 통합 ────────────────────────
    brk_enhanced = round(
        brk * 0.5 + brk_persist * 0.5, 1  # 기존 돌파 + 지속력 평균
    )

    # ── 재정의 risk: 무조건 감점 아닌 필터 역할 ─────────────────────
    risk_combined = round(
        risk * 0.4 + risk_filter * 0.6, 1  # 기존 risk + 새 risk_filter 혼합
    )

    sect_str = _sector_strength(sector_avg_ret20)
    haircut  = (regime or {}).get("kr_haircut",0)

    # 한국 전용 최적 가중치 로드 (미국 가중치와 분리)
    w = None
    try:
        from screener.backtest import load_optimal_weights_kr
        w = load_optimal_weights_kr()
        w_lead = w.get("leader",   0.33)
        w_vol  = w.get("volume",   0.18)
        w_brk  = w.get("breakout", 0.10)
        w_cat  = w.get("catalyst", 0.12)
        w_qot  = w.get("quality",  0.08)
        w_sup  = 0.12
        w_sect = max(0.03, 1.0 - w_lead - w_vol - w_brk - w_cat - w_qot - w_sup)
        w_buy  = w.get("buyable",  0.28)
        w_mom  = w.get("momentum", 0.22)
        w_risk = w.get("risk_penalty", 0.10)
    except Exception:
        w = {"leader":0.33,"volume":0.18,"breakout":0.10,"catalyst":0.12,"quality":0.08,"risk_penalty":0.10}
        w_lead=0.33; w_vol=0.18; w_brk=0.10; w_cat=0.12
        w_qot=0.08;  w_sup=0.12; w_sect=0.07
        w_buy=0.28;  w_mom=0.22; w_risk=0.10

    # quality 팩터 제거 — 백테스트에서 수익률·급등 적중 양쪽 모두 역효과(-)로 판정됨.
    # 캐시된 가중치까지 덮어 강제 0으로 고정 (3개 점수 경로 모두 반영).
    w = {**w, "quality": 0.0}
    w_qot = 0.0

    # leader_final: breakout 포함 (미국과 동일 철학)
    leader_final  = round(max(0,min(100,
        ldr      * w_lead +
        vol      * w_vol  +
        brk      * w_brk  +
        sect_str * w_sect +
        sup      * w_sup  +
        cat      * w_cat  +
        qot      * w_qot
    ))*(1-haircut),1)

    # buyable_final: buyable 강하게, supply 포함
    buyable_final = round(max(0,min(100,
        mom      * w_mom  +
        buy      * w_buy  +
        vol      * w_vol  +
        sup      * w_sup  +
        qot      * w_qot  +
        cat      * 0.08   +
        sect_str * 0.05   -
        risk     * w_risk
    ))*(1-haircut),1)

    track    = classify_track(ldr, buy, brk, risk, mom)
    risk_flag = ""
    if item.get("is_warned"):              risk_flag = "투자경고"
    elif risk >= 65:                       risk_flag = "추격위험"
    elif _has_big_bearish(df,thr=-0.06,w=3): risk_flag = "장대음봉"
    elif _has_long_upper_wick(df,w=3):     risk_flag = "긴윗꼬리"

    row = {
        "ticker":         ticker, "name": item.get("name",ticker),
        "market":         mkt,    "sector": item.get("sector",""),
        "price":          round(item.get("price",float("nan")),0),
        "market_cap":     mc,     "avg_amount": avg_amt,
        "ret_5d":         round(r5,2),  "ret_20d": round(r20,2), "ret_60d": round(r60,2),
        "rs_mkt_20":      round(rs_mkt_20,2),  "rs_mkt_60":    round(rs_mkt_60,2),
        "rs_sector_20":   round(rs_sect_20,2), "rs_rank_pct":  round(rs_rank_pct*100,1),
        "vol_surge":      round(surge,3) if math.isfinite(surge) else float("nan"),
        "foreign_5d":     item.get("foreign_5d",float("nan")),
        "foreign_20d":    item.get("foreign_20d",float("nan")),
        "inst_5d":        item.get("inst_5d",float("nan")),
        "inst_20d":       item.get("inst_20d",float("nan")),
        "supply_score":   round(sup,1),
        "week52_prox":    round(prox52,2) if math.isfinite(prox52) else float("nan"),
        "week52_low_prox":round(_week52_low_prox(df),2),
        "above_20dma":    above20, "above_60dma": above60, "above_200dma": above200,
        "dist_20dma":     round(dist20,2) if math.isfinite(dist20) else float("nan"),
        "dist_60dma":     round(dist60,2) if math.isfinite(dist60) else float("nan"),
        "ann_vol":        round(ann_vol,2) if math.isfinite(ann_vol) else float("nan"),
        # 기존 factor (원본 보존)
        "leader_score":   ldr,  "momentum_score": mom, "volume_score":   vol,
        "buyable_score":  buy,  "breakout_score": brk, "top_risk_score": risk,
        "quality_score":  qot,  "catalyst_score": cat, "supply_score":   round(sup,1),
        # 신규 factor (급등주 탐지 강화)
        "momentum_accel": mom_accel,
        "volume_shock":   vol_shock,
        "brk_persist":    brk_persist,
        "near_high_20d":  near_hi["near_20d"],
        "near_high_60d":  near_hi["near_60d"],
        "near_high_120d": near_hi["near_120d"],
        "near_high":      near_hi_comp,
        "vol_contract":   vol_contract,
        "pullback_q":     pullback_q,
        "risk_filter":    risk_filter,
        "sect_rs":        sect_rs,        # 섹터 상대강도 (독립)
        "supply_flow":    sup_flow,       # 수급 흐름 연속성 (독립)
        # 재정의 factor (enhanced)
        "volume_enhanced":  vol_enhanced,
        "breakout_enhanced": brk_enhanced,
        "risk_combined":    risk_combined,
        "leader_final":   leader_final,  "buyable_final": buyable_final,
        "final_score":    leader_final,
        "track":          track, "risk_flag": risk_flag,
        "naver_link":     f"https://finance.naver.com/item/main.naver?code={ticker}",
        "news_link":      f"https://search.naver.com/search.naver?where=news&query={item.get('name','')}",
        "dart_link":      f"https://dart.fss.or.kr/dsab001/main.do?autoSearch=true&textCrpNm={item.get('name','')}",
        "kind_link":      f"https://kind.krx.co.kr/corpgeneral/corpNamePopup.do?method=searchCorpName&searchText={item.get('name','')}",
    }
    row["summary"] = build_summary(row, "KR")

    # validated_score: enhanced + 신규 factor 기반 + KR 최적 가중치 반영
    if _cvs_series is not None:
        _score_row = {
            "leader":             row.get("leader_score", 50),
            "volume":             row.get("volume_score", 50),
            "breakout":           row.get("breakout_score", 50),
            "catalyst":           row.get("catalyst_score", 50),
            "quality":            row.get("quality_score", 50),
            "risk":               row.get("top_risk_score", 30),
            "ret_20d":            row.get("ret_20d", 0),
            "volume_enhanced":    row.get("volume_enhanced", row.get("volume_score", 50)),
            "breakout_enhanced":  row.get("breakout_enhanced", row.get("breakout_score", 50)),
            "risk_combined":      row.get("risk_combined", row.get("top_risk_score", 30)),
            "momentum_accel":     row.get("momentum_accel", 50),
            "near_high":          row.get("near_high", 50),
            "vol_contract":       row.get("vol_contract", 50),
            "pullback_q":         row.get("pullback_q", 50),
            "brk_persist":        row.get("brk_persist", 50),
            "sect_rs":            row.get("sect_rs", 50),
            "supply_flow":        row.get("supply_flow", 50),
        }
        row["validated_score"] = float(_cvs_series(pd.DataFrame([_score_row]), weights=w).iloc[0])
    elif _cvs is not None:
        row["validated_score"] = _cvs(
            leader   = float(row.get("leader_score",   50) or 50),
            volume   = float(row.get("volume_enhanced",
                             row.get("volume_score", 50)) or 50),
            breakout = float(row.get("breakout_enhanced",
                             row.get("breakout_score", 50)) or 50),
            catalyst = float(row.get("catalyst_score", 50) or 50),
            quality  = float(row.get("quality_score",  50) or 50),
            risk     = float(row.get("risk_combined",
                             row.get("top_risk_score", 30)) or 30),
            ret_20d  = float(row.get("ret_20d",         0) or 0),
            weights  = w,   # KR 최적 가중치 반영
        )
    else:
        row["validated_score"] = row.get("leader_score", 50)

    return row


# ════════════════════════════════════════════════════════════════════
# 신규 Factor 함수 — 국내 급등주 탐지 최적화
# ════════════════════════════════════════════════════════════════════

def momentum_accel_score(close: pd.Series) -> float:
    """
    모멘텀 가속: 최근 5일 수익률이 20일 수익률 대비 가속되는지.
    급등 초기 신호 포착에 중요.
    """
    if len(close) < 22: return 50.0
    try:
        r5  = _ret(close, 5)
        r20 = _ret(close, 20)
        r20_daily = r20 / 20 if r20 != 0 else 0
        r5_daily  = r5  / 5  if r5  != 0 else 0

        if not (math.isfinite(r5) and math.isfinite(r20)): return 50.0

        # 가속 비율
        accel = r5_daily - r20_daily
        score = 50.0
        if accel > 1.0:   score += 35   # 강한 가속
        elif accel > 0.5: score += 20
        elif accel > 0.2: score += 10
        elif accel < -1.0: score -= 25  # 감속
        elif accel < -0.3: score -= 10

        # 단기가 양수면서 장기도 양수: 추세 속 가속
        if r5 > 0 and r20 > 0: score += 10
        if r5 > 5 and r20 > 0: score += 5
        return round(max(0, min(100, score)), 1)
    except Exception:
        return 50.0


def volume_shock_score(df: pd.DataFrame) -> float:
    """
    거래대금 충격: 최근 3~5일 거래대금이 20일 평균 대비 몇 배인지.
    가격 상승 동반 여부도 반영.
    """
    if "Volume" not in df.columns or len(df) < 22: return 50.0
    try:
        close  = df["Close"]
        vol    = df["Volume"]
        amount = (close * vol).fillna(0)
        if amount.empty: return 50.0

        avg20 = float(amount.tail(25).head(20).mean())
        avg5  = float(amount.tail(5).mean())
        avg3  = float(amount.tail(3).mean())
        if avg20 <= 0: return 50.0

        ratio5 = avg5 / avg20
        ratio3 = avg3 / avg20

        score = 50.0
        # 거래대금 급증
        if ratio3 >= 3.0:   score += 35
        elif ratio3 >= 2.0: score += 25
        elif ratio3 >= 1.5: score += 15
        elif ratio3 >= 1.2: score += 8
        elif ratio3 < 0.5:  score -= 15

        # 5일 지속성
        if ratio5 >= 2.0:   score += 10
        elif ratio5 >= 1.5: score += 5

        # 가격 상승 동반 여부
        if len(close) >= 5:
            r5 = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100 if float(close.iloc[-6]) > 0 else 0
            if ratio3 >= 1.5 and r5 > 0:  score += 10   # 거래량+상승 동반
            if ratio3 >= 1.5 and r5 < -3: score -= 15   # 거래량+하락: 매도세

        return round(max(0, min(100, score)), 1)
    except Exception:
        return 50.0


def breakout_persistence_score(close: pd.Series, df: pd.DataFrame) -> float:
    """
    돌파 지속력: 신고가/박스권 돌파 후 가격을 유지하는지.
    단순 돌파 여부가 아니라 돌파 후 유지력 평가.
    """
    if len(close) < 25: return 50.0
    try:
        score = 50.0
        cur   = float(close.iloc[-1])
        hi20  = float(close.tail(21).iloc[:-1].max()) if len(close) >= 21 else cur
        hi60  = float(close.tail(61).iloc[:-1].max()) if len(close) >= 61 else cur

        # 20일 신고가 돌파
        if cur > hi20 * 1.01:
            score += 20
            # 3일 전도 신고가 위였는지 (지속성 확인)
            if len(close) >= 4 and float(close.iloc[-3]) > hi20 * 0.98:
                score += 10
        elif cur > hi20 * 0.97:
            score += 8  # 신고가 근처

        # 60일 신고가 돌파
        if cur > hi60 * 1.01:
            score += 15
            if len(close) >= 6 and float(close.iloc[-5]) > hi60 * 0.97:
                score += 10  # 5일 이상 유지

        # 눌림 깊이 확인 (돌파 후 얼마나 빠지지 않았는지)
        if len(close) >= 10:
            recent_low  = float(close.tail(5).min())
            recent_high = float(close.tail(10).max())
            if recent_high > 0:
                pullback_depth = (recent_high - recent_low) / recent_high * 100
                if pullback_depth < 3:   score += 10   # 아주 얕은 눌림
                elif pullback_depth < 7: score += 5
                elif pullback_depth > 15: score -= 10  # 깊은 눌림

        # 음봉 연속 패턴 (지속력 약화)
        if len(close) >= 3:
            if float(close.iloc[-1]) < float(close.iloc[-2]) and \
               float(close.iloc[-2]) < float(close.iloc[-3]):
                score -= 15  # 3일 연속 음봉

        return round(max(0, min(100, score)), 1)
    except Exception:
        return 50.0


def near_high_score(df: pd.DataFrame) -> dict:
    """
    고점 근접도: 20일/60일/120일 고점 근접도.
    반환: {"near_20d": float, "near_60d": float, "near_120d": float, "composite": float}
    """
    result = {"near_20d": 50.0, "near_60d": 50.0, "near_120d": 50.0, "composite": 50.0}
    try:
        if "High" not in df.columns or len(df) < 5: return result
        cur = float(df["Close"].iloc[-1])
        scores = []
        for days, key in [(20, "near_20d"), (60, "near_60d"), (120, "near_120d")]:
            n = min(days, len(df))
            hi = float(df["High"].tail(n).max())
            if hi <= 0: continue
            prox = (cur / hi) * 100
            # 0~100 변환: 95%+ 근접 시 고점
            if prox >= 98:    s = 90
            elif prox >= 95:  s = 75
            elif prox >= 90:  s = 60
            elif prox >= 80:  s = 45
            elif prox >= 70:  s = 30
            else:             s = max(0, prox - 20)
            result[key] = round(s, 1); scores.append(s)
        if scores:
            # 120일 근접 가중 평균 (장기 신고가 근접이 더 중요)
            weights = [0.2, 0.35, 0.45][:len(scores)]
            result["composite"] = round(sum(s*w for s,w in zip(scores,weights)) / sum(weights[:len(scores)]), 1)
    except Exception:
        pass
    return result


def volatility_contraction_score(close: pd.Series, df: pd.DataFrame) -> float:
    """
    변동성 수축 후 거래량 증가: VCP (Volatility Contraction Pattern) 탐지.
    변동성이 줄어들다가 거래량 터지는 패턴 = 강한 돌파 신호.
    """
    if len(close) < 30: return 50.0
    try:
        # 변동성 수축 측정
        vol_recent = float(close.tail(5).pct_change().std() * 100)  if len(close) >= 6  else float("nan")
        vol_mid    = float(close.tail(20).pct_change().std() * 100) if len(close) >= 21 else float("nan")
        vol_long   = float(close.tail(60).pct_change().std() * 100) if len(close) >= 61 else float("nan")

        score = 50.0
        if math.isfinite(vol_recent) and math.isfinite(vol_mid):
            contraction = vol_mid / vol_recent if vol_recent > 0 else 1.0
            if contraction >= 2.0:   score += 25   # 변동성 강한 수축
            elif contraction >= 1.5: score += 15
            elif contraction >= 1.2: score += 8
            elif contraction < 0.8:  score -= 10   # 변동성 확대 중

        if math.isfinite(vol_mid) and math.isfinite(vol_long):
            mid_contraction = vol_long / vol_mid if vol_mid > 0 else 1.0
            if mid_contraction >= 1.5: score += 10

        # 변동성 수축 중 거래량 증가 (VCP 핵심)
        if "Volume" not in df.columns: return round(max(0,min(100,score)),1)
        vol_surge = _dv_surge(df)
        if math.isfinite(vol_surge) and math.isfinite(vol_recent) and math.isfinite(vol_mid):
            if vol_surge >= 1.5 and contraction >= 1.3:
                score += 20  # VCP 신호!
            elif vol_surge >= 1.2 and contraction >= 1.2:
                score += 10

        return round(max(0, min(100, score)), 1)
    except Exception:
        return 50.0


def pullback_quality_score(close: pd.Series, df: pd.DataFrame) -> float:
    """
    눌림 질: 돌파 후 눌림이 얕고 거래량이 줄어드는지.
    좋은 눌림 = 작은 하락 + 거래량 감소 → 다음 상승 준비.
    """
    if len(close) < 20: return 50.0
    try:
        score = 50.0
        # 20일 고점 대비 현재 위치
        hi20 = float(close.tail(20).max())
        cur  = float(close.iloc[-1])
        if hi20 <= 0: return 50.0

        pullback = (hi20 - cur) / hi20 * 100
        if pullback < 3:    score += 20   # 고점 바로 아래
        elif pullback < 7:  score += 12
        elif pullback < 12: score += 5
        elif pullback > 20: score -= 20   # 깊은 조정
        elif pullback > 15: score -= 10

        # 눌림 중 거래량 감소 여부
        if "Volume" not in df.columns: return round(max(0,min(100,score)),1)
        vol5  = float(df["Volume"].tail(5).mean()) if len(df) >= 5 else 0
        vol20 = float(df["Volume"].tail(20).mean()) if len(df) >= 20 else 0
        if vol20 > 0:
            vol_ratio = vol5 / vol20
            if 3 < pullback < 15 and vol_ratio < 0.8:  score += 15  # 눌림+거래량감소=좋은눌림
            elif 3 < pullback < 15 and vol_ratio > 1.3: score -= 10  # 눌림+거래량증가=약점

        # 양봉 캔들 패턴 (반등 시작)
        if len(close) >= 3:
            if float(close.iloc[-1]) > float(close.iloc[-2]):
                if float(close.iloc[-2]) < float(close.iloc[-3]):  # 반등 캔들
                    score += 8

        return round(max(0, min(100, score)), 1)
    except Exception:
        return 50.0


def risk_filter_score(close: pd.Series, df: pd.DataFrame) -> float:
    """
    위험 필터 (재정의): 무조건 감점하지 않고, 
    저유동성/급락/관리종목성 위험만 강하게 감점.
    건강한 상승 시 패널티 없음.
    """
    if len(close) < 10: return 30.0
    try:
        score = 20.0  # 기본값 낮게 (패널티 위주 설계)

        # 급락 위험 (3일 내 -10% 이상)
        r3 = (float(close.iloc[-1]) / float(close.iloc[-4]) - 1) * 100 if len(close) >= 4 else 0
        if r3 < -15:    score += 60  # 매우 강한 급락
        elif r3 < -10:  score += 40
        elif r3 < -5:   score += 20

        # 저유동성 위험 (거래대금 극도로 낮음)
        if "Volume" not in df.columns or len(df) < 5:
            pass
        else:
            close_s = df["Close"]
            vol_s   = df["Volume"]
            amount  = close_s * vol_s
            avg5    = float(amount.tail(5).mean())
            # 1억 미만 거래대금: 저유동성 위험
            if avg5 < 100_000_000:   score += 50
            elif avg5 < 500_000_000: score += 25

        # 연속 급등 후 윗꼬리 위험
        if _has_long_upper_wick(df, w=3): score += 20
        if _has_big_bearish(df, thr=-0.07, w=2): score += 30

        # 가격 극단 상승 (단기 과열, MU식에서도 일정 부분 감점)
        r10 = _ret(close, 10) if len(close) >= 11 else 0
        if math.isfinite(r10) and r10 > 50: score += 20  # 10일 +50% 이상
        elif math.isfinite(r10) and r10 > 30: score += 10

        return round(min(100, max(0, score)), 1)
    except Exception:
        return 30.0


def sector_relative_strength_score(
    rs_sector_20: float,
    rs_rank_pct:  float = 0.5,
    sector_avg_ret20: float = float("nan"),
    sector_avg_ret60: float = float("nan"),
) -> float:
    """
    섹터 상대강도: 같은 섹터/테마 내 상대강도 독립 factor.
    기존 leader_score에 섞여 있던 섹터 신호를 별도로 분리.
    """
    score = 50.0
    try:
        if math.isfinite(rs_sector_20):
            if rs_sector_20 >= 10:    score += 30
            elif rs_sector_20 >= 5:   score += 20
            elif rs_sector_20 >= 2:   score += 10
            elif rs_sector_20 < -10:  score -= 25
            elif rs_sector_20 < -5:   score -= 15

        if math.isfinite(rs_rank_pct):
            pct = rs_rank_pct if rs_rank_pct <= 1 else rs_rank_pct / 100
            if pct >= 0.9:   score += 15
            elif pct >= 0.8: score += 10
            elif pct >= 0.7: score += 5
            elif pct <= 0.3: score -= 10
            elif pct <= 0.2: score -= 15

        # 섹터 평균 대비 개별 종목 추가강도
        if math.isfinite(sector_avg_ret20) and math.isfinite(rs_sector_20):
            # 섹터 자체가 강할 때 그 종목이 섹터보다 더 강하면 더 좋음
            if sector_avg_ret20 > 5 and rs_sector_20 > 5: score += 10
            # 섹터가 약한데 혼자 강하면 단기 과매수 위험
            if sector_avg_ret20 < -5 and rs_sector_20 > 10: score -= 5

    except Exception:
        pass
    return round(max(0, min(100, score)), 1)


def supply_flow_score(
    foreign_5d:  float,
    foreign_20d: float,
    inst_5d:     float,
    inst_20d:    float,
    foreign_consecutive: int = 0,
    inst_consecutive:    int = 0,
) -> float:
    """
    수급 흐름 연속성: 외국인/기관 순매수 연속성 중심 factor.
    기존 kr_supply_score를 확장해 연속성과 방향 전환 신호를 강화.
    """
    score = 50.0
    try:
        def _safe(x):
            try: return float(x) if x is not None else float("nan")
            except: return float("nan")

        f5  = _safe(foreign_5d);  f20 = _safe(foreign_20d)
        i5  = _safe(inst_5d);     i20 = _safe(inst_20d)

        # 외국인 수급
        if math.isfinite(f5) and math.isfinite(f20):
            if f5 > 0 and f20 > 0:   score += 20   # 단기+중기 모두 순매수
            elif f5 > 0 and f20 < 0: score += 8    # 최근 전환 매수
            elif f5 < 0 and f20 > 0: score -= 5    # 최근 전환 매도
            elif f5 < 0 and f20 < 0: score -= 15   # 지속 매도

        # 기관 수급
        if math.isfinite(i5) and math.isfinite(i20):
            if i5 > 0 and i20 > 0:   score += 15
            elif i5 > 0 and i20 < 0: score += 5
            elif i5 < 0 and i20 < 0: score -= 10

        # 연속 매수 일수 (데이터 있으면)
        if foreign_consecutive > 0:
            if foreign_consecutive >= 5:   score += 15
            elif foreign_consecutive >= 3: score += 8
        elif foreign_consecutive < 0:
            if foreign_consecutive <= -5:  score -= 15
            elif foreign_consecutive <= -3: score -= 8

        if inst_consecutive > 0:
            if inst_consecutive >= 5:      score += 10
            elif inst_consecutive >= 3:    score += 5
        elif inst_consecutive < 0:
            if inst_consecutive <= -5:     score -= 10

        # 쌍매수 (외국인+기관 동시 순매수): 가장 강한 신호
        f_pos = math.isfinite(f5) and f5 > 0
        i_pos = math.isfinite(i5) and i5 > 0
        if f_pos and i_pos:
            score += 15

    except Exception:
        pass
    return round(max(0, min(100, score)), 1)
