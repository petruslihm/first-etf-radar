"""
팩터 함수 특성화(회귀) 테스트.
calc_us_factors / calc_kr_factors 의 출력이 골든 스냅샷과 동일한지 검증한다.
공통 헬퍼 추출 리팩터가 동작을 바꾸지 않았음을 보장하고,
앞으로 두 함수를 의도치 않게 바꾸면 여기서 실패한다.

(의도적으로 팩터 로직을 바꿨다면:
   python3 tools/char_factors.py capture  로 골든 재생성)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.char_factors import _us_inputs, _kr_inputs, _normalize, GOLDEN


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("골든 스냅샷 없음 — tools/char_factors.py capture 필요")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _collect_diffs(golden_rows, current_rows, market):
    diffs = []
    for i, (g, c) in enumerate(zip(golden_rows, current_rows)):
        for k in set(g) | set(c):
            if g.get(k, "__MISSING__") != c.get(k, "__MISSING__"):
                diffs.append(f"[{market}#{i}] {k}: {g.get(k)} → {c.get(k)}")
    return diffs


def test_us_factors_unchanged(golden):
    cur = _normalize(_us_inputs())
    diffs = _collect_diffs(golden["us"], cur, "us")
    assert not diffs, "calc_us_factors 출력이 골든과 다름:\n" + "\n".join(diffs[:20])


def test_kr_factors_unchanged(golden):
    cur = _normalize(_kr_inputs())
    diffs = _collect_diffs(golden["kr"], cur, "kr")
    assert not diffs, "calc_kr_factors 출력이 골든과 다름:\n" + "\n".join(diffs[:20])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
