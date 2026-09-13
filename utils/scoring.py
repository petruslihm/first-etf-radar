"""
utils/scoring.py
백테스트 + 실전 스크리너가 공유하는 단일 점수 공식.

calc_validated_score()를 중심으로 backtest.py / factors.py / engine.py 가
모두 같은 공식을 사용해야 합니다.
"""

import math


# ── 기본 가중치 ──────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "leader":       0.38,
    "volume":       0.18,
    "breakout":     0.12,
    "catalyst":     0.15,
    "quality":      0.00,   # 제거: 백테스트에서 수익률·급등 적중 양쪽 역효과
    "risk_penalty": 0.08,
}


def calc_risk_penalty(risk: float, ret_20d: float = 0.0) -> float:
    """
    Risk 패널티 계산.
    철학: 과열 자체는 감점 없음. risk >= 90 AND ret_20d < 0 일 때만 강하게.
    """
    if not math.isfinite(risk): risk = 30.0
    if not math.isfinite(ret_20d): ret_20d = 0.0
    if risk >= 90 and ret_20d < 0:  return 15.0
    if risk >= 90:                   return 8.0
    if risk >= 80:                   return 2.0
    return 0.0


def calc_validated_score(
    leader:    float,
    volume:    float,
    breakout:  float,
    catalyst:  float,
    quality:   float,
    risk:      float = 30.0,
    ret_20d:   float = 0.0,
    weights:   dict  = None,
) -> float:
    """
    백테스트와 실전 스크리너가 공유하는 단일 점수 공식.

    포함 요소 (백테스트로 검증 가능):
      leader, volume, breakout, catalyst, quality, risk_penalty

    제외 요소 (final_score 보정용으로 분리):
      supply_score, sector_strength, fund_score, analyst_score, theme_boost

    양수 가중치는 합이 1.0이 되도록 자동 정규화.
    risk_penalty는 감점 계수이므로 정규화 제외.
    """
    w    = weights or DEFAULT_WEIGHTS
    pen  = calc_risk_penalty(risk, ret_20d)

    # 양수 가중치 정규화 (합 = 1.0)
    pos_keys = ["leader", "volume", "breakout", "catalyst", "quality"]
    pos_sum  = sum(w.get(k, DEFAULT_WEIGHTS[k]) for k in pos_keys)
    if pos_sum <= 0: pos_sum = 1.0
    norm = 1.0 / pos_sum

    score = (
        float(leader)   * w.get("leader",   DEFAULT_WEIGHTS["leader"])   * norm +
        float(volume)   * w.get("volume",   DEFAULT_WEIGHTS["volume"])   * norm +
        float(breakout) * w.get("breakout", DEFAULT_WEIGHTS["breakout"]) * norm +
        float(catalyst) * w.get("catalyst", DEFAULT_WEIGHTS["catalyst"]) * norm +
        float(quality)  * w.get("quality",  DEFAULT_WEIGHTS["quality"])  * norm -
        pen             * w.get("risk_penalty", DEFAULT_WEIGHTS["risk_penalty"])
    )
    return round(float(score), 1)

def calc_validated_score_series(df, weights: dict = None):
    """
    calc_validated_score()와 같은 공식을 pandas Series 단위로 계산합니다.
    신규 factor 포함: enhanced 컬럼이 있으면 우선 사용.
      volume   → volume_enhanced   (없으면 volume)
      breakout → breakout_enhanced (없으면 breakout)
      risk     → risk_combined     (없으면 risk)

    신규 factor 독립 가중치도 지원:
      momentum_accel, near_high, vol_contract, pullback_q, brk_persist, sect_rs, supply_flow
    """
    import numpy as _np
    import pandas as _pd

    w = weights or DEFAULT_WEIGHTS
    idx = df.index

    def _col(name, default, fallback=None):
        # enhanced 컬럼 우선, 없으면 원본, 없으면 default
        for n in ([name] if fallback is None else [name, fallback]):
            if n in df.columns:
                return _pd.to_numeric(df[n], errors="coerce").fillna(default)
        return _pd.Series(default, index=idx, dtype="float64")

    leader   = _col("leader",   50.0)
    # enhanced 컬럼 우선
    volume   = _col("volume_enhanced",   50.0, "volume")
    breakout = _col("breakout_enhanced", 50.0, "breakout")
    catalyst = _col("catalyst", 50.0)
    quality  = _col("quality",  50.0)
    risk     = _col("risk_combined", 30.0, "risk")
    ret_20d  = _col("ret_20d",   0.0)

    # 신규 factor 독립 가중치 지원
    mom_accel  = _col("momentum_accel", 50.0)
    near_high  = _col("near_high",      50.0)
    vol_cont   = _col("vol_contract",   50.0)
    pullback_q = _col("pullback_q",     50.0)
    brk_persist= _col("brk_persist",    50.0)
    sect_rs    = _col("sect_rs",        50.0)
    supply_flow= _col("supply_flow",    50.0)

    pen = _np.where((risk >= 90) & (ret_20d < 0), 15.0,
          _np.where(risk >= 90, 8.0,
          _np.where(risk >= 80, 2.0, 0.0)))

    # 양수 가중치 정규화 (신규 factor 포함)
    pos_keys_base = ["leader", "volume", "breakout", "catalyst", "quality"]
    pos_keys_new  = ["momentum_accel", "near_high", "vol_contract", "pullback_q", "brk_persist", "sect_rs", "supply_flow"]
    pos_sum = sum(float(w.get(k, DEFAULT_WEIGHTS.get(k, 0)) or 0) for k in pos_keys_base + pos_keys_new)
    norm = 1.0 / pos_sum if pos_sum > 0 else 1.0

    score = (
        leader   * float(w.get("leader",   DEFAULT_WEIGHTS["leader"])   or 0) * norm +
        volume   * float(w.get("volume",   DEFAULT_WEIGHTS["volume"])   or 0) * norm +
        breakout * float(w.get("breakout", DEFAULT_WEIGHTS["breakout"]) or 0) * norm +
        catalyst * float(w.get("catalyst", DEFAULT_WEIGHTS["catalyst"]) or 0) * norm +
        quality  * float(w.get("quality",  DEFAULT_WEIGHTS["quality"])  or 0) * norm +
        # 신규 factor (가중치 0이면 영향 없음)
        mom_accel  * float(w.get("momentum_accel", 0) or 0) * norm +
        near_high  * float(w.get("near_high",      0) or 0) * norm +
        vol_cont   * float(w.get("vol_contract",   0) or 0) * norm +
        pullback_q * float(w.get("pullback_q",     0) or 0) * norm +
        brk_persist* float(w.get("brk_persist",    0) or 0) * norm +
        sect_rs    * float(w.get("sect_rs",        0) or 0) * norm +
        supply_flow* float(w.get("supply_flow",    0) or 0) * norm -
        _pd.Series(pen, index=idx) * float(w.get("risk_penalty", DEFAULT_WEIGHTS["risk_penalty"]) or 0)
    )
    return score.astype(float).round(1)

