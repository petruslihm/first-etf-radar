"""저널 파싱·채점 테스트 (네트워크 불필요)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from screener import journal


def test_parse_with_fence_and_prose():
    txt = """분석 결과입니다:
```json
{"as_of":"2026-06-04","market":"KR","picks":[
  {"rank":1,"ticker":"089970","name":"브이엠","action":"진입","stop_price":65000,"stop_pct":-6}
]}
```
끝."""
    out = journal.parse_gpt_json(txt)
    assert out["as_of"] == "2026-06-04"
    assert len(out["picks"]) == 1
    assert out["picks"][0]["ticker"] == "089970"
    assert out["picks"][0]["stop_price"] == 65000


def test_parse_bare_json():
    out = journal.parse_gpt_json('{"picks":[{"ticker":"005930","action":"관망"}]}')
    assert out["picks"][0]["ticker"] == "005930"
    assert out["picks"][0]["rank"] == 1  # 누락 시 인덱스로 보정


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        journal.parse_gpt_json("이건 JSON이 아님")
    with pytest.raises(ValueError):
        journal.parse_gpt_json('{"picks":[]}')


def test_verdict_logic():
    assert journal._verdict("진입", 5.0, False) == "✅성공"
    assert journal._verdict("진입", 1.0, True) == "❌손절"   # 손절 우선
    assert journal._verdict("진입", -5.0, False) == "❌부진"
    assert journal._verdict("진입", 0.5, False) == "➖중립"
    assert journal._verdict("회피", 12.0, False) == "⚠️놓침"
    assert journal._verdict("관망", -6.0, False) == "✅회피적중"


def test_score_no_lookahead():
    """채점은 진입일 '이후' 가격만 사용 — 진입일 이전을 바꿔도 결과 불변."""
    idx = pd.date_range("2026-05-01", periods=30, freq="B")
    prices = [100.0] * 30
    df = pd.DataFrame({"Open": prices, "High": [p*1.01 for p in prices],
                       "Low": [p*0.99 for p in prices], "Close": prices,
                       "Volume": [1e6]*30}, index=idx)
    entry_dt = idx[10]
    e1 = journal._eval_after(df, entry_dt, 5)
    # 진입일 이전 가격을 폭등시켜도 평가구간(이후)은 불변
    df2 = df.copy(); df2.iloc[:11, :] = 9999.0
    e2 = journal._eval_after(df2, entry_dt, 5)
    assert e1[0] == e2[0] and abs(e1[1] - e2[1]) < 1e-9


def test_eval_after_insufficient_returns_none():
    idx = pd.date_range("2026-05-01", periods=12, freq="B")
    df = pd.DataFrame({"Close": [100.0]*12, "Low": [99.0]*12}, index=idx)
    # 진입일이 끝에서 2번째 → 5일 평가 불가
    assert journal._eval_after(df, idx[10], 5) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
