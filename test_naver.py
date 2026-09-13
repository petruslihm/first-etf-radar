"""네이버 금융으로 한국 전종목 수집 테스트"""
import requests, pandas as pd, time, re
from io import StringIO
from bs4 import BeautifulSoup

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://finance.naver.com/"
})

def get_all_tickers(market="KOSPI"):
    sosok = "0" if market == "KOSPI" else "1"
    url = "https://finance.naver.com/sise/sise_market_sum.naver"

    # 마지막 페이지 파악
    r = s.get(url, params={"sosok": sosok, "page": "1"}, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    pager = soup.select("td.pgRR > a")
    last_page = int(pager[0]["href"].split("page=")[1]) if pager else 1
    print(f"{market} 페이지 수: {last_page}")

    codes, names = [], []
    for page in range(1, last_page + 1):
        r = s.get(url, params={"sosok": sosok, "page": page}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a[href*='code=']"):
            code = a["href"].split("code=")[1][:6]
            if code.isdigit() and len(code) == 6:
                codes.append(code)
                names.append(a.text.strip())
        time.sleep(0.15)
        if page % 5 == 0:
            print(f"  {page}/{last_page} 페이지... ({len(codes)}개)")

    return list(zip(codes, names))

def get_ohlcv(code, pages=26):
    """네이버 일봉 OHLCV (1페이지=10일, 26페이지≈252일)"""
    url = "https://finance.naver.com/item/sise_day.naver"
    rows = []
    for page in range(1, pages + 1):
        r = s.get(url, params={"code": code, "page": page}, timeout=10)
        tbls = pd.read_html(StringIO(r.text), encoding="utf-8")
        for t in tbls:
            cols = [str(c) for c in t.columns]
            if any(k in " ".join(cols) for k in ["날짜","종가","시가","고가","저가","거래량"]):
                df = t.dropna(how="all").dropna(subset=[t.columns[0]])
                rows.append(df)
                break
        time.sleep(0.05)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True).drop_duplicates()
    return df

# ── 테스트 ────────────────────────────────────────────────────────
print("=== 1. KOSPI 전종목 코드 수집 ===")
kospi = get_all_tickers("KOSPI")
print(f"KOSPI: {len(kospi)}개")
print("예시:", kospi[:5])

print("\n=== 2. KOSDAQ 전종목 코드 수집 ===")
kosdaq = get_all_tickers("KOSDAQ")
print(f"KOSDAQ: {len(kosdaq)}개")

print("\n=== 3. 삼성전자 OHLCV ===")
df = get_ohlcv("005930", pages=3)
if df is not None:
    print(f"컬럼: {list(df.columns)}")
    print(df.head(5))
else:
    print("실패")
