"""
팩터 헬퍼 + 트랙 분류 + look-ahead 슬라이싱 불변식 테스트.
네트워크/외부 데이터 없이 순수 함수만 검증한다.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.factors import _clamp, _rs, _ret, classify_track


# ── 순수 수학 헬퍼 ────────────────────────────────────────────────────
def test_clamp_basic():
    assert _clamp(150, 0, 100) == 100
    assert _clamp(-20, 0, 100) == 0
    assert _clamp(50, 0, 100) == 50

def test_clamp_nan_returns_neutral():
    assert _clamp(float("nan"), 0, 100, neutral=42) == 42
    assert _clamp(None, 0, 100, neutral=7) == 7

def test_rs_relative_strength():
    assert _rs(10, 4) == 6.0
    assert math.isnan(_rs(float("nan"), 4))

def test_ret_pct_change():
    close = pd.Series([100, 101, 102, 103, 104, 110])  # 6 points
    # _ret(close, 5): (110/100 - 1)*100 = 10%
    assert abs(_ret(close, 5) - 10.0) < 1e-6

def test_ret_insufficient_history():
    close = pd.Series([100, 101])
    assert math.isnan(_ret(close, 20))


# ── classify_track: 경계값에서 의도한 트랙이 나오는지 ─────────────────────
def test_track_hot_leader_extended():
    # leader>=75, momentum>=60, risk>=65 → Extended
    assert classify_track(80, 50, 50, risk=70, momentum=65) == "Hot Leader / Extended"

def test_track_hot_leader_buyable():
    # leader>=75, momentum>=60, risk<65 → Buyable
    assert classify_track(80, 50, 50, risk=40, momentum=65) == "Hot Leader / Buyable"

def test_track_breakout_signal():
    assert classify_track(50, 40, 70, risk=50, momentum=50) == "Breakout Signal"

def test_track_avoid_weak_momentum():
    assert classify_track(40, 40, 40, risk=50, momentum=10) == "Avoid / Weak"

def test_track_watch_only():
    # leader>=50 이지만 다른 조건 미충족
    assert classify_track(52, 40, 40, risk=50, momentum=40) == "Watch Only"


# ── look-ahead 불변식: T 시점 슬라이스는 미래를 포함하면 안 된다 ──────────
def test_lookahead_slice_excludes_future():
    """
    백테스트의 핵심 안전장치: df.iloc[:t_idx+1] 는 t_idx 이후를 절대 포함하지 않는다.
    이 불변식이 깨지면 백테스트가 미래를 훔쳐본다.
    """
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"Close": np.arange(100, 200)}, index=idx)
    t_idx = 60
    sliced = df.iloc[:t_idx + 1]
    assert len(sliced) == t_idx + 1
    assert sliced.index.max() == idx[t_idx]
    # 미래 데이터(>t_idx)가 단 하나도 없어야 함
    assert (sliced.index <= idx[t_idx]).all()
    assert df.iloc[t_idx + 1:].index.min() > sliced.index.max()


def test_lookahead_date_filter_invariant():
    """스냅샷 방식(index<=날짜)도 동일하게 미래 차단."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"Close": np.arange(100, 200)}, index=idx)
    cutoff = idx[60]
    sliced = df[df.index <= cutoff]
    assert sliced.index.max() == cutoff
    assert not (sliced.index > cutoff).any()


def test_supply_flow_at_no_lookahead():
    """
    수급 점수가 결정일 이후 데이터를 절대 쓰지 않는지 검증.
    결정일 이후 수급을 극단값으로 바꿔도 T시점 점수가 변하면 안 된다.
    """
    from screener.backtest import _supply_flow_at
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    base = pd.DataFrame({"foreign": np.full(60, 1000.0),
                         "inst":    np.full(60, 500.0)}, index=idx)
    T = idx[40]
    s1 = _supply_flow_at(base, T, mc=1e11, avg_amt=1e10)
    # 미래(>T) 수급을 거대 음수로 오염
    poisoned = base.copy()
    poisoned.loc[idx[41]:, "foreign"] = -1e9
    poisoned.loc[idx[41]:, "inst"]    = -1e9
    s2 = _supply_flow_at(poisoned, T, mc=1e11, avg_amt=1e10)
    assert s1 == s2, f"미래 수급이 T시점 점수에 샜다: {s1} vs {s2}"


def test_supply_flow_at_empty_returns_neutral():
    from screener.backtest import _supply_flow_at
    assert _supply_flow_at(None, pd.Timestamp("2024-01-01")) == 50.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
