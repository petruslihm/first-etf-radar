#!/usr/bin/env python3
"""
수급(외국인/기관) 과거 데이터 깊이 진단.

백테스트에 수급을 넣으려면, 네이버 frgn 페이지가 '과거 날짜별' 순매매를
얼마나 깊게 주는지부터 알아야 한다(백테스트 기간 ~2년을 덮는지).
이 스크립트는 종목 몇 개에 대해 frgn 페이지를 여러 장 긁어,
날짜별 외국인/기관 순매매 시계열을 만들고 '며칠치/몇 개월치'인지 보고한다.

사용:
  python3 tools/probe_supply.py                  # 기본 종목들
  python3 tools/probe_supply.py 005930 000660    # 특정 종목
"""
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from screener.collector import _nav, _get_with_retry

FRGN_URL = "https://finance.naver.com/item/frgn.naver"
MAX_PAGES_PROBE = 40   # 한 종목당 최대 몇 페이지까지 시도할지 (1페이지 ≈ 20거래일)


def _find_cols(df: pd.DataFrame):
    """frgn 테이블에서 (날짜, 외국인순매매, 기관순매매) 컬럼을 찾는다."""
    date_col = fgn_col = inst_col = None
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            top, bot = str(col[0]), str(col[1])
            if "날짜" in top or "날짜" in bot:
                date_col = col
            if "외국인" in top and "순매매" in bot:
                fgn_col = col
            if "기관" in top and "순매매" in bot:
                inst_col = col
    else:
        for c in df.columns:
            cs = str(c)
            if "날짜" in cs: date_col = c
            elif "외국인" in cs and "순매" in cs: fgn_col = c
            elif "기관" in cs and "순매" in cs: inst_col = c
    return date_col, fgn_col, inst_col


def _num(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.replace("+", "").str.strip(),
        errors="coerce")


def probe_one(code: str) -> dict:
    s = _nav()
    rows = []
    pages_with_data = 0
    for page in range(1, MAX_PAGES_PROBE + 1):
        r = _get_with_retry(s, FRGN_URL, params={"code": code, "page": page}, retries=1)
        if r is None:
            break
        try:
            tbls = pd.read_html(StringIO(r.text), encoding="utf-8")
        except Exception:
            break
        got = False
        for t in tbls:
            dcol, fcol, icol = _find_cols(t)
            if dcol is None or (fcol is None and icol is None):
                continue
            sub = t[[c for c in [dcol, fcol, icol] if c is not None]].copy()
            sub.columns = ["date"] + (["fgn"] if fcol is not None else []) + (["inst"] if icol is not None else [])
            sub["date"] = pd.to_datetime(sub["date"].astype(str).str.strip(),
                                         errors="coerce", format="%Y.%m.%d")
            sub = sub.dropna(subset=["date"])
            if len(sub):
                if "fgn" in sub:  sub["fgn"] = _num(sub["fgn"])
                if "inst" in sub: sub["inst"] = _num(sub["inst"])
                rows.append(sub); got = True
                break
        if got:
            pages_with_data += 1
        else:
            break  # 더 이상 데이터 페이지 없음

    if not rows:
        return {"code": code, "ok": False}
    alld = pd.concat(rows).drop_duplicates(subset=["date"]).sort_values("date")
    return {
        "code": code, "ok": True,
        "pages": pages_with_data, "n_days": len(alld),
        "oldest": alld["date"].min().date(), "newest": alld["date"].max().date(),
        "span_days": (alld["date"].max() - alld["date"].min()).days,
        "has_fgn": "fgn" in alld.columns and alld["fgn"].notna().any(),
        "has_inst": "inst" in alld.columns and alld["inst"].notna().any(),
    }


def main():
    codes = sys.argv[1:] or ["005930", "000660", "247540", "086520"]  # 삼성전자/하이닉스/에코프로비엠/에코프로
    print(f"수급 과거 데이터 깊이 진단 (종목당 최대 {MAX_PAGES_PROBE}페이지 시도)\n")
    print(f"{'종목':>8}{'페이지':>7}{'거래일수':>9}{'가장오래된날':>14}{'최근날':>13}{'기간(일)':>9}  외인/기관")
    results = []
    for c in codes:
        r = probe_one(c)
        results.append(r)
        if not r["ok"]:
            print(f"{c:>8}   ❌ 데이터 파싱 실패")
            continue
        print(f"{c:>8}{r['pages']:>7}{r['n_days']:>9}{str(r['oldest']):>14}"
              f"{str(r['newest']):>13}{r['span_days']:>9}   {r['has_fgn']}/{r['has_inst']}")

    ok = [r for r in results if r.get("ok")]
    if ok:
        max_span = max(r["span_days"] for r in ok)
        print(f"\n최대 확보 기간: 약 {max_span}일 (≈ {max_span//30}개월)")
        print("판정:")
        if max_span >= 540:
            print("  ✅ 2년 가까이 확보 — 백테스트 전 구간 수급 편입 가능")
        elif max_span >= 250:
            print("  🟡 약 1년 — 최근 1년 구간에 한해 수급 백테스트 가능")
        else:
            print("  🔴 너무 짧음 — 과거 수급 백테스트 불가. 라이브 전용으로만 사용하거나")
            print("     별도 수급 데이터 소스(유료 API 등)가 필요.")


if __name__ == "__main__":
    main()
