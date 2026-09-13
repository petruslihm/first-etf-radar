"""
특성화 하니스: calc_us_factors / calc_kr_factors 의 출력을 합성 데이터로 박제한다.
리팩터 전/후 출력이 완전히 동일한지 비교하기 위한 골든 스냅샷 생성·검증 도구.

사용:
  python3 tools/char_factors.py capture   # 골든 저장 (리팩터 전)
  python3 tools/char_factors.py verify    # 골든과 비교 (리팩터 후)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOLDEN = Path(__file__).resolve().parent / "_golden_factors.json"


def _ohlcv(seed: int, n: int = 260) -> pd.DataFrame:
    """결정적 합성 OHLCV (상승추세 + 노이즈)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    drift = np.linspace(0, 0.6, n)
    noise = rng.normal(0, 0.015, n).cumsum()
    close = 100 * np.exp(drift + noise)
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    openp = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": openp, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def _bench(seed: int, n: int = 260) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 400 * np.exp(np.linspace(0, 0.2, n) + rng.normal(0, 0.008, n).cumsum())
    return pd.Series(close, index=idx)


def _us_inputs():
    from screener.factors import calc_us_factors
    cases = []
    for seed in (1, 7, 42):
        df = _ohlcv(seed)
        item = {
            "ohlcv": df, "company": f"Co{seed}", "sector": "Technology",
            "industry": "Software", "price": float(df["Close"].iloc[-1]),
            "market_cap": 5e9 + seed * 1e9, "avg_dv": 1e8,
            "earnings_date": "2024-02-01",
        }
        bench = {"SPY": _bench(100), "QQQ": _bench(101), "XLK": _bench(102)}
        row = calc_us_factors(
            f"TST{seed}", item, bench, "XLK",
            sector_avg_ret20=8.0, sector_avg_ret60=15.0,
            rs_rank_pct=0.6 + seed * 0.01, regime={"us_haircut": 0.0},
        )
        cases.append(row)
    return cases


def _kr_inputs():
    from screener.factors import calc_kr_factors
    cases = []
    for seed in (3, 11, 99):
        df = _ohlcv(seed)
        item = {
            "ohlcv": df, "name": f"종목{seed}", "market": "KOSPI",
            "sector": "반도체", "price": float(df["Close"].iloc[-1]),
            "market_cap": 3e11 + seed * 1e10, "avg_amount": 5e10,
            "foreign_5d": 1e5, "foreign_20d": 3e5,
            "inst_5d": 5e4, "inst_20d": 1e5,
            "foreign_consecutive": 3, "inst_consecutive": 2,
            "is_warned": False,
        }
        bench = {"kospi": _bench(200), "kosdaq": _bench(201)}
        row = calc_kr_factors(
            f"00{seed}90", item, bench,
            sector_avg_ret20=6.0, rs_rank_pct=0.55 + seed * 0.001,
            regime={"kr_haircut": 0.0},
        )
        cases.append(row)
    return cases


def _normalize(obj):
    """NaN을 문자열로, float를 라운딩해 JSON 안정 비교."""
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__NaN__"
        return round(obj, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _normalize(float(obj))
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def capture():
    data = {"us": _normalize(_us_inputs()), "kr": _normalize(_kr_inputs())}
    GOLDEN.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"✅ 골든 저장: {GOLDEN}  (US {len(data['us'])}건, KR {len(data['kr'])}건)")


def verify():
    if not GOLDEN.exists():
        print("❌ 골든 파일 없음 — 먼저 capture 실행"); sys.exit(1)
    golden = json.loads(GOLDEN.read_text())
    current = {"us": _normalize(_us_inputs()), "kr": _normalize(_kr_inputs())}
    diffs = []
    for market in ("us", "kr"):
        for i, (g, c) in enumerate(zip(golden[market], current[market])):
            keys = set(g) | set(c)
            for k in keys:
                if g.get(k, "__MISSING__") != c.get(k, "__MISSING__"):
                    diffs.append(f"[{market}#{i}] {k}: {g.get(k)} → {c.get(k)}")
    if diffs:
        print(f"❌ 출력 불일치 {len(diffs)}건:")
        for d in diffs[:40]:
            print("   ", d)
        sys.exit(1)
    print(f"✅ 리팩터 전후 출력 100% 동일 (US+KR 전 필드 일치)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"capture": capture, "verify": verify}.get(cmd, verify)()
