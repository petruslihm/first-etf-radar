"""
WSL에서 실행: python3 check_fundamental.py
yfinance + 네이버에서 가져올 수 있는 재무 데이터 확인
"""
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from bs4 import BeautifulSoup

print("=" * 60)
print("1. yfinance info (미국 NVDA)")
print("=" * 60)
obj  = yf.Ticker("NVDA")
info = obj.info or {}
keys = [
    'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months',
    'profitMargins', 'operatingMargins', 'revenueGrowth', 'earningsGrowth',
    'returnOnEquity', 'debtToEquity', 'earningsQuarterlyGrowth',
    'totalRevenue', 'revenuePerShare', 'trailingEps', 'forwardEps',
    'pegRatio', 'enterpriseToRevenue',
]
for k in keys:
    v = info.get(k)
    if v is not None:
        print(f"  {k}: {v}")

print()
print("=" * 60)
print("2. yfinance quarterly_income_stmt")
print("=" * 60)
try:
    df = obj.quarterly_income_stmt
    if not df.empty:
        print(f"행: {df.index.tolist()[:8]}")
        print(f"컬럼(분기): {list(df.columns)}")
        print(df.iloc[:5, :4])
    else:
        print("비어있음")
except Exception as e:
    print(f"오류: {e}")

print()
print("=" * 60)
print("3. 네이버 삼성전자(005930) 재무 페이지")
print("=" * 60)
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://finance.naver.com/"})

# 네이버 재무 페이지
r = s.get("https://finance.naver.com/item/coinfo.naver",
          params={"code": "005930", "target": "finsum_tab4"}, timeout=10)
print(f"상태: {r.status_code}")
try:
    tbls = pd.read_html(StringIO(r.text), encoding="utf-8")
    print(f"테이블 수: {len(tbls)}")
    for i, t in enumerate(tbls):
        if len(t) > 2:
            print(f"\n테이블{i}: shape={t.shape}")
            print(f"컬럼: {list(t.columns)[:6]}")
            print(t.head(5).to_string())
except Exception as e:
    print(f"파싱 오류: {e}")

# 네이버 실적 페이지
print("\n=== 네이버 실적(finsum) ===")
r2 = s.get("https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx",
           params={"cmp_cd": "005930"}, timeout=10)
print(f"wisereport 상태: {r2.status_code}")
