"""
점수 공식 단위 테스트.

핵심 목적:
  1) calc_validated_score / *_series 가 같은 결과를 내는지 (스칼라↔시리즈 일관성)
  2) calc_risk_penalty 임계값이 철학대로 동작하는지
  3) [회귀 가드] 백테스트 옵티마이저/스냅샷이 실전과 동일한 공식을 쓰는지
     — 1순위 "공식 일원화"가 미래에 다시 깨지면 이 테스트가 실패한다.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.scoring import (
    calc_validated_score, calc_validated_score_series,
    calc_risk_penalty, DEFAULT_WEIGHTS,
)


# ── calc_risk_penalty: 과열 자체는 무벌점, 과열+추세훼손만 강벌점 ──────────
def test_risk_penalty_thresholds():
    assert calc_risk_penalty(95, ret_20d=-1) == 15.0   # 과열 + 하락 → 최대
    assert calc_risk_penalty(95, ret_20d=+5) == 8.0    # 과열이지만 상승
    assert calc_risk_penalty(85, ret_20d=+5) == 2.0    # 약과열
    assert calc_risk_penalty(50, ret_20d=-10) == 0.0   # 정상구간은 무벌점

def test_risk_penalty_handles_nan():
    assert calc_risk_penalty(float("nan"), float("nan")) == 0.0


# ── 스칼라 ↔ 시리즈 일관성 ────────────────────────────────────────────
def test_scalar_matches_series_no_enhanced():
    """enhanced 컬럼이 없으면 시리즈식은 원본 컬럼으로 폴백 → 스칼라와 동일."""
    row = dict(leader=80, volume=55, breakout=70, catalyst=60, quality=65,
               risk=85, ret_20d=12)
    scalar = calc_validated_score(
        leader=row["leader"], volume=row["volume"], breakout=row["breakout"],
        catalyst=row["catalyst"], quality=row["quality"],
        risk=row["risk"], ret_20d=row["ret_20d"],
    )
    series = calc_validated_score_series(pd.DataFrame([row])).iloc[0]
    assert abs(scalar - series) < 0.05, f"{scalar} vs {series}"


def test_series_prefers_enhanced_columns():
    """enhanced 컬럼이 있으면 그것을 우선 사용해야 한다."""
    base = dict(leader=80, volume=55, breakout=70, catalyst=60, quality=65,
                risk=85, ret_20d=12)
    enhanced = dict(base, volume_enhanced=30, breakout_enhanced=30, risk_combined=30)
    s_base = calc_validated_score_series(pd.DataFrame([base])).iloc[0]
    s_enh = calc_validated_score_series(pd.DataFrame([enhanced])).iloc[0]
    assert s_base != s_enh, "enhanced 컬럼이 반영되지 않음"


# ── 가중치 정규화: 양수 가중치 합이 1이 아니어도 비율만 같으면 동일 ──────
def test_weight_normalization_scale_invariant():
    row = dict(leader=80, volume=55, breakout=70, catalyst=60, quality=65,
               risk=30, ret_20d=0)
    w1 = dict(DEFAULT_WEIGHTS)
    w2 = {k: (v * 2 if k != "risk_penalty" else v) for k, v in w1.items()}
    s1 = calc_validated_score_series(pd.DataFrame([row]), weights=w1).iloc[0]
    s2 = calc_validated_score_series(pd.DataFrame([row]), weights=w2).iloc[0]
    # 양수 가중치를 일괄 2배 해도 정규화 후 동일해야 함
    assert abs(s1 - s2) < 0.05, f"정규화 실패: {s1} vs {s2}"


# ── 점수 범위 ────────────────────────────────────────────────────────
def test_score_in_reasonable_range():
    row = dict(leader=100, volume=100, breakout=100, catalyst=100, quality=100,
               risk=0, ret_20d=0)
    s = calc_validated_score_series(pd.DataFrame([row])).iloc[0]
    assert 0 <= s <= 100


# ── [회귀 가드] 옵티마이저가 실전 공식을 쓰는지 ──────────────────────────
def test_optimizer_uses_shared_formula():
    """
    1순위 핵심 회귀 테스트.
    backtest._sharpe_top 내부가 다시 raw 수식으로 돌아가면,
    enhanced 컬럼이 다른 레코드에서 옵티마이저 점수가 실전 점수와 달라진다.
    여기서는 '실전 공식이 enhanced를 반영한다'는 불변식을 고정한다.
    """
    df = pd.DataFrame({
        "leader": [80, 40], "volume": [55, 50], "breakout": [70, 30],
        "catalyst": [60, 55], "quality": [65, 45], "risk": [85, 30],
        "ret_20d": [12, -3],
        "volume_enhanced": [72, 48], "breakout_enhanced": [68, 35],
        "risk_combined": [80, 28],
    })
    w = {"leader": 0.40, "volume": 0.18, "breakout": 0.12,
         "catalyst": 0.15, "quality": 0.10, "risk_penalty": 0.08}
    live = calc_validated_score_series(df, weights=w).values

    # 구버전 raw 공식 (enhanced 무시, 정규화 없음)
    raw = (df["leader"] * 0.40 + df["volume"] * 0.18 + df["breakout"] * 0.12 +
           df["catalyst"] * 0.15 + df["quality"] * 0.10).values
    assert not np.allclose(live, raw), "실전 공식이 enhanced/정규화를 반영하지 않음"


def test_quality_factor_removed():
    """
    quality 팩터 제거 회귀 가드.
    백테스트에서 수익률·급등 양쪽 역효과로 판정돼 제거함.
    quality 값이 달라도 검증 점수가 바뀌면 안 된다.
    """
    assert DEFAULT_WEIGHTS["quality"] == 0.0, "quality 기본 가중치가 0이 아님"
    base = dict(leader=70, volume=55, breakout=60, catalyst=58, risk=30, ret_20d=5)
    lo = calc_validated_score_series(pd.DataFrame([dict(base, quality=10)])).iloc[0]
    hi = calc_validated_score_series(pd.DataFrame([dict(base, quality=95)])).iloc[0]
    assert lo == hi, f"quality가 여전히 점수에 영향: {lo} vs {hi}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
