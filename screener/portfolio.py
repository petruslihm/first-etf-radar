"""
보유 종목 / 실제 매수 추적 모듈.

목적:
  1) 사용자가 '실제로 보유 중인 종목'을 기록 → 스크리너 선별에 무조건 포함,
     GPT 프롬프트에도 자동으로 들어감.
  2) GPT가 추천(순위)한 것과 별개로, 사용자가 '실제로 매수한' 종목을 체크.
  3) 저장 위치는 zip 바깥(~/.etf-radar-cache/portfolio.json) — 코드 갱신/재시작과
     무관하게 보존. WSL 껐다 켜도 유지.

데이터 구조 (portfolio.json):
{
  "holdings": {            # 보유 종목
    "005930": {"name": "삼성전자", "avg_price": 70000, "qty": 10,
                "memo": "...", "added": "2026-06-12"},
    ...
  },
  "bought": {             # 실제 매수 체크 (저널 pick 중 실제 산 것)
    "KR:2026-06-12:089970": {"bought": true, "real_price": 70000,
                              "memo": "...", "ts": "..."},
    ...
  }
}
"""
import json
from datetime import date
from typing import Optional

from screener.paths import DATA_DIR

_PF_PATH = DATA_DIR / "portfolio.json"


# ── 로드/세이브 ───────────────────────────────────────────────────────
def _load() -> dict:
    if not _PF_PATH.exists():
        return {"holdings": {}, "bought": {}}
    try:
        d = json.loads(_PF_PATH.read_text(encoding="utf-8"))
        d.setdefault("holdings", {})
        d.setdefault("bought", {})
        return d
    except Exception:
        return {"holdings": {}, "bought": {}}


def _save(data: dict):
    try:
        _PF_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── 보유 종목 (holdings) ─────────────────────────────────────────────
def get_holdings() -> dict:
    """보유 종목 dict 반환. {ticker: {name, avg_price, qty, memo, added}}"""
    return _load().get("holdings", {})


def get_holding_tickers() -> list[str]:
    """보유 종목 티커 리스트 (스크리너 강제 포함용)."""
    return list(_load().get("holdings", {}).keys())


def add_holding(ticker: str, name: str = "", avg_price=None,
                qty=None, memo: str = "", weight_raw: str = ""):
    """보유 종목 추가/수정."""
    ticker = str(ticker).strip().upper()
    if not ticker:
        return
    data = _load()
    data["holdings"][ticker] = {
        "name": str(name).strip(),
        "avg_price": _num(avg_price),
        "qty": _num(qty),
        "weight_raw": str(weight_raw).strip(),
        "memo": str(memo).strip(),
        "added": data["holdings"].get(ticker, {}).get("added") or date.today().isoformat(),
    }
    _save(data)


def remove_holding(ticker: str):
    ticker = str(ticker).strip().upper()
    data = _load()
    if ticker in data["holdings"]:
        del data["holdings"][ticker]
        _save(data)


def set_holdings_bulk(rows: list[dict]):
    """여러 보유 종목을 한 번에 저장 (UI 데이터에디터 저장용).
    rows: [{ticker, name, avg_price, qty, memo}, ...]"""
    data = _load()
    new_h = {}
    for r in rows:
        tk = str(r.get("ticker", "")).strip().upper()
        if not tk:
            continue
        new_h[tk] = {
            "name": str(r.get("name", "")).strip(),
            "avg_price": _num(r.get("avg_price")),
            "qty": _num(r.get("qty")),
            "weight_raw": str(r.get("weight_raw", "") or "").strip(),
            "memo": str(r.get("memo", "")).strip(),
            "added": data["holdings"].get(tk, {}).get("added") or date.today().isoformat(),
        }
    data["holdings"] = new_h
    _save(data)


# ── 실제 매수 체크 (bought) ──────────────────────────────────────────
def _bought_key(market: str, d: str, ticker: str) -> str:
    return f"{market}:{d}:{str(ticker).strip().upper()}"


def is_bought(market: str, d: str, ticker: str) -> bool:
    key = _bought_key(market, d, ticker)
    rec = _load().get("bought", {}).get(key)
    return bool(rec and rec.get("bought"))


def get_bought_record(market: str, d: str, ticker: str) -> dict:
    key = _bought_key(market, d, ticker)
    return _load().get("bought", {}).get(key, {})


def set_bought(market: str, d: str, ticker: str, bought: bool = True,
               real_price=None, memo: str = ""):
    """저널 pick 종목을 '실제 매수함'으로 체크."""
    data = _load()
    key = _bought_key(market, d, ticker)
    if bought:
        data["bought"][key] = {
            "bought": True,
            "real_price": _num(real_price),
            "memo": str(memo).strip(),
            "ts": date.today().isoformat(),
        }
    else:
        data["bought"].pop(key, None)
    _save(data)


# ── 유틸 ─────────────────────────────────────────────────────────────
def _num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(str(v).replace(",", "").strip())
        return f
    except Exception:
        return None


def parse_weight(raw) -> Optional[float]:
    """비중 입력을 0~1 실수로 변환.
    - '3/5' 같은 분수 → 0.6
    - '0.6' 또는 '60%' → 0.6
    - 빈 값/오류 → None
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if s.endswith("%"):
            return float(s[:-1].strip()) / 100.0
        if "/" in s:
            num, den = s.split("/", 1)
            num, den = float(num.strip()), float(den.strip())
            if den == 0:
                return None
            return num / den
        v = float(s)
        # 1보다 크면 퍼센트로 입력한 것으로 간주 (예: 50 → 0.5)
        return v / 100.0 if v > 1 else v
    except Exception:
        return None


def compute_allocation(holdings: dict) -> dict:
    """보유 종목들의 비중을 계산하고 현금을 자동 산출.
    반환: {
      "rows": [{ticker, name, weight_raw, weight, weight_pct}, ...],
      "invested": 0.85,        # 종목 합
      "cash": 0.15,            # 1 - invested (음수면 0)
      "cash_pct": 15.0,
      "over": False,           # 합이 1을 초과하면 True (경고용)
    }
    """
    rows = []
    invested = 0.0
    for tk, h in holdings.items():
        w = parse_weight(h.get("weight_raw"))
        wv = w if w is not None else 0.0
        invested += wv
        rows.append({
            "ticker": tk,
            "name": h.get("name", ""),
            "weight_raw": h.get("weight_raw", ""),
            "weight": wv,
            "weight_pct": round(wv * 100, 1),
        })
    over = invested > 1.0 + 1e-9
    cash = max(0.0, 1.0 - invested)
    return {
        "rows": rows,
        "invested": round(invested, 4),
        "cash": round(cash, 4),
        "cash_pct": round(cash * 100, 1),
        "over": over,
    }


# ── 종목 검색 (UI 자동완성/등록 확인용) ──────────────────────────────
def build_name_map(market: str = "KR") -> dict:
    """{티커: 종목명} 매핑. 한국은 kr_universe의 종목명 dict 사용.
    미국은 종목명 소스가 없어 티커 목록만 등록(이름은 티커로 대체).
    app에서 세션 full_df의 name과 합쳐 더 풍부하게 만들 수 있음."""
    name_map = {}
    if market == "KR":
        try:
            import data.kr_universe as ku
            for k, v in vars(ku).items():
                if isinstance(v, dict) and not k.startswith("_"):
                    for code, name in v.items():
                        if isinstance(code, str) and isinstance(name, str) and code.isdigit():
                            name_map[code] = name
        except Exception:
            pass
    elif market == "US":
        try:
            import data.us_universe as uu
            # 티커 리스트들을 모아 등록 (이름은 없으므로 티커=이름)
            for k, v in vars(uu).items():
                if k.startswith("_"):
                    continue
                if isinstance(v, list):
                    for t in v:
                        if isinstance(t, str) and t.replace("-", "").isalpha():
                            name_map.setdefault(t.upper(), t.upper())
                elif isinstance(v, dict):
                    for t in v.keys():
                        if isinstance(t, str) and t.replace("-", "").isalpha():
                            name_map.setdefault(t.upper(), t.upper())
        except Exception:
            pass
    return name_map


def build_name_map_all() -> dict:
    """한국+미국 통합 매핑 (시장 구분 없이 검색용)."""
    nm = build_name_map("KR")
    nm.update(build_name_map("US"))
    return nm


def search_stocks(query: str, name_map: dict, limit: int = 20) -> list:
    """질의어로 종목 검색 (티커 또는 종목명 부분일치).
    반환: [(ticker, name), ...]"""
    q = str(query).strip().upper()
    if not q:
        return []
    hits = []
    for tk, nm in name_map.items():
        if q in tk.upper() or q in str(nm).upper():
            hits.append((tk, nm))
            if len(hits) >= limit:
                break
    return hits
