"""
수급 페이지 구조 진단 스크립트
WSL에서 실행: python3 diagnose_supply.py
"""
import requests, pandas as pd, time
from io import StringIO
from bs4 import BeautifulSoup

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://finance.naver.com/"
})

code = "005930"  # 삼성전자

print("=" * 60)
print("1. frgn.naver (외국인 매매동향)")
print("=" * 60)
r = s.get("https://finance.naver.com/item/frgn.naver",
          params={"code": code}, timeout=10)
print(f"상태: {r.status_code}, 길이: {len(r.text)}")
try:
    tbls = pd.read_html(StringIO(r.text), encoding="utf-8")
    for i, t in enumerate(tbls):
        print(f"\n테이블{i}: shape={t.shape}")
        print(f"컬럼: {list(t.columns)}")
        print(t.head(5).to_string())
except Exception as e:
    print(f"파싱 실패: {e}")

time.sleep(0.5)

print("\n" + "=" * 60)
print("2. investor.naver (투자자별 매매동향 - 외국인+기관 같이)")
print("=" * 60)
r2 = s.get("https://finance.naver.com/item/investor.naver",
           params={"code": code}, timeout=10)
print(f"상태: {r2.status_code}, 길이: {len(r2.text)}")
try:
    tbls2 = pd.read_html(StringIO(r2.text), encoding="utf-8")
    for i, t in enumerate(tbls2):
        if len(t) > 3:
            print(f"\n테이블{i}: shape={t.shape}")
            print(f"컬럼: {list(t.columns)}")
            print(t.head(5).to_string())
except Exception as e:
    print(f"파싱 실패: {e}")

time.sleep(0.5)

print("\n" + "=" * 60)
print("3. sise_day.naver 거래대금 있는지 확인")
print("=" * 60)
r3 = s.get("https://finance.naver.com/item/sise_day.naver",
           params={"code": code, "page": 1}, timeout=10)
try:
    tbls3 = pd.read_html(StringIO(r3.text), encoding="utf-8")
    for i, t in enumerate(tbls3):
        if len(t) > 3:
            print(f"테이블{i} 컬럼: {list(t.columns)}")
            print(t.head(3).to_string())
except Exception as e:
    print(f"파싱 실패: {e}")
