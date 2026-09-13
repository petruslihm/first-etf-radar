"""
기록/검증 저널 — GPT가 뽑은 '단기 대응' 후보를 날짜별로 저장하고,
며칠 뒤 실제 가격으로 맞았는지 자동 채점한다.

설계 요지(요구 4제약 대응):
  1) 빠진 날 대응: 채점은 '다음날'에 묶지 않는다. 각 기록에 기준일을 박아두고,
     나중에 아무 때나 열면 그 사이 실제 가격으로 소급 채점한다.
  2) 업데이트 보존: 저장 위치는 zip 바깥(~/.etf-radar-cache/journal). 코드 갱신과 무관.
  3) 종목수 유동: picks 배열 — 몇 개든 저장/채점.
  4) 쉬운 입력/비교: GPT가 고정 JSON으로 뱉게 하고, 붙여넣기→저장. 표로 비교.

채점은 look-ahead 없이: 기준일 D의 종가를 진입가로, D 이후(엄격히 초과) N거래일째
종가를 평가가로 쓴다. 손절선은 그 구간 저가가 손절선을 깼는지로 판정.
"""
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from screener.paths import DATA_DIR

JOURNAL_DIR = DATA_DIR / "journal"
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DAYS_DEFAULT = 5          # 채점 시 진입 후 며칠(거래일) 뒤 종가로 평가
WIN_PCT = 3.0                  # 진입: +3% 이상 성공
MISS_PCT = 10.0               # 관망/회피: +10% 이상 올랐으면 '놓침'
AVOID_OK_PCT = -5.0            # 관망/회피: -5% 이하면 '잘 피함'


# ── 저장/조회 ────────────────────────────────────────────────────────
def _path(market: str, d: str) -> Path:
    return JOURNAL_DIR / f"picks_{market}_{d}.json"


def list_dates(market: str) -> list[str]:
    out = []
    for f in JOURNAL_DIR.glob(f"picks_{market}_*.json"):
        m = re.search(r"_(\d{4}-\d{2}-\d{2})\.json$", f.name)
        if m:
            out.append(m.group(1))
    return sorted(out, reverse=True)


def load(market: str, d: str) -> Optional[dict]:
    p = _path(market, d)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save(market: str, d: str, entry: dict):
    entry["market"] = market
    entry["as_of"] = d
    entry["saved_at"] = datetime.now().isoformat(timespec="seconds")
    _path(market, d).write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")


def delete(market: str, d: str) -> bool:
    p = _path(market, d)
    if p.exists():
        p.unlink()
        return True
    return False


# ── GPT JSON 파싱 (관대하게) ─────────────────────────────────────────
def parse_gpt_json(text: str) -> dict:
    """
    GPT가 준 텍스트에서 JSON을 추출. ```json 펜스/앞뒤 설명이 섞여도 처리.
    반환: {"as_of":..., "picks":[...]} (필드 누락은 기본값 보정).
    실패 시 ValueError.
    """
    if not text or not text.strip():
        raise ValueError("빈 입력")

    raw = text.strip()
    # 1) ```json ... ``` 펜스 우선
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    candidate = m.group(1).strip() if m else raw
    # 2) 첫 { 부터 마지막 } 까지
    if not candidate.lstrip().startswith("{"):
        s, e = candidate.find("{"), candidate.rfind("}")
        if s != -1 and e != -1 and e > s:
            candidate = candidate[s:e+1]

    try:
        data = json.loads(candidate)
    except Exception as ex:
        raise ValueError(f"JSON 파싱 실패: {ex}")

    picks_raw = data.get("picks") or data.get("종목") or []
    if not isinstance(picks_raw, list) or not picks_raw:
        raise ValueError("picks 배열이 없음")

    picks = []
    for i, p in enumerate(picks_raw, 1):
        if not isinstance(p, dict):
            continue
        tk = str(p.get("ticker") or p.get("티커") or "").strip()
        if not tk:
            continue
        picks.append({
            "rank":      _int(p.get("rank", i), i),
            "ticker":    tk,
            "name":      str(p.get("name") or p.get("종목") or tk),
            "action":    str(p.get("action") or p.get("대응") or "관망"),
            "trigger":   str(p.get("trigger") or p.get("트리거") or ""),
            "stop_price": _num(p.get("stop_price") or p.get("손절가")),
            "stop_pct":  _num(p.get("stop_pct") or p.get("손절률")),
            "target_price": _num(p.get("target_price") or p.get("목표가")),
            "target_pct":_num(p.get("target_pct") or p.get("목표률")),
            "note":      str(p.get("note") or p.get("판단") or ""),
        })
    if not picks:
        raise ValueError("유효한 종목이 없음")
    return {"as_of": data.get("as_of") or data.get("date"), "picks": picks}


def _int(v, default):
    try: return int(v)
    except Exception: return default

def _num(v):
    try:
        f = float(v)
        return f
    except Exception:
        return None


# ── OHLCV 조회 (per-stock 캐시에서) ──────────────────────────────────
def _load_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    from screener.paths import CACHE_DIR
    cands = sorted(CACHE_DIR.glob(f"kr_ohlcv_{ticker}_*p.pkl"))
    if not cands:
        return None
    try:
        df = pd.read_pickle(cands[-1])
        if isinstance(df, pd.DataFrame) and "Close" in df.columns:
            return df.sort_index()
    except Exception:
        pass
    return None


def _close_on_or_before(df: pd.DataFrame, d) -> Optional[tuple]:
    """기준일 d 이하의 마지막 종가 → (date, close)."""
    sub = df[df.index <= pd.Timestamp(d)]
    if len(sub) == 0:
        return None
    return sub.index[-1], float(sub["Close"].iloc[-1])


def _eval_after(df: pd.DataFrame, entry_dt, n: int):
    """entry_dt '초과' n번째 거래일의 (date, close, min_low, max_high). 부족하면 None."""
    fut = df[df.index > pd.Timestamp(entry_dt)]
    if len(fut) < n:
        return None
    window = fut.iloc[:n]
    ev_dt = window.index[-1]
    ev_close = float(window["Close"].iloc[-1])
    low_col = "Low" if "Low" in window.columns else "Close"
    high_col = "High" if "High" in window.columns else "Close"
    min_low = float(window[low_col].min())
    max_high = float(window[high_col].max())
    return ev_dt, ev_close, min_low, max_high


# ── 진입가 캡처 (저장 시) ────────────────────────────────────────────
def capture_entry_prices(picks: list[dict], as_of: str) -> dict:
    """기준일(as_of) 종가를 진입 기준가로 박제. {ticker: {entry_date, entry_close}}."""
    out = {}
    for p in picks:
        df = _load_ohlcv(p["ticker"])
        if df is None:
            continue
        r = _close_on_or_before(df, as_of)
        if r:
            out[p["ticker"]] = {"entry_date": str(r[0].date()), "entry_close": r[1]}
    return out


# ── 채점 ─────────────────────────────────────────────────────────────
def score_entry(entry: dict, eval_days: int = EVAL_DAYS_DEFAULT) -> list[dict]:
    """
    각 pick을 현재 캐시된 가격으로 채점. look-ahead 없음(진입일 '이후'만).
    반환: pick + 채점 결과(rows). 가격/데이터 부족 시 status='대기'.
    """
    as_of = entry.get("as_of")
    entry_prices = entry.get("entry_prices", {})
    rows = []
    for p in entry.get("picks", []):
        tk = p["ticker"]
        row = dict(p)
        df = _load_ohlcv(tk)
        ep = entry_prices.get(tk)
        # 진입가: 저장 시 박제값 우선, 없으면 OHLCV에서 as_of 종가
        entry_close = entry_date = None
        if ep:
            entry_close = ep.get("entry_close"); entry_date = ep.get("entry_date")
        elif df is not None and as_of:
            r = _close_on_or_before(df, as_of)
            if r: entry_date, entry_close = str(r[0].date()), r[1]

        if df is None or entry_close is None or entry_date is None:
            row.update(status="데이터없음", ret=None, verdict="—")
            rows.append(row); continue

        ev = _eval_after(df, entry_date, eval_days)
        # 목표 업사이드는 평가 전에도 계산 가능(진입가 + 목표가/목표%)
        _tp = p.get("target_price")
        if _tp is not None:
            _tu = (float(_tp) / entry_close - 1) * 100
        elif p.get("target_pct") is not None:
            _tu = float(p["target_pct"])
        else:
            _tu = None
        if ev is None:
            row.update(status=f"대기({eval_days}일미경과)", entry_close=round(entry_close,1),
                       entry_date=entry_date, ret=None,
                       target_up=round(_tu,1) if _tu is not None else None,
                       verdict="⏳")
            rows.append(row); continue

        ev_dt, ev_close, min_low, max_high = ev
        ret = (ev_close / entry_close - 1) * 100
        stop_hit = (p.get("stop_price") is not None
                    and min_low <= float(p["stop_price"]))
        # 목표 업사이드: 목표가 우선, 없으면 목표% 사용
        tgt_price = p.get("target_price")
        if tgt_price is not None:
            target_up = (float(tgt_price) / entry_close - 1) * 100
        elif p.get("target_pct") is not None:
            target_up = float(p["target_pct"])
        else:
            target_up = None
        target_hit = (tgt_price is not None and max_high >= float(tgt_price))
        verdict = _verdict(p.get("action", "관망"), ret, stop_hit)
        # ── 매도 타이밍: 최신 종가 기준 '지금' 신호 (보유 관리용) ──
        now_close = float(df["Close"].iloc[-1])
        now_ret = (now_close / entry_close - 1) * 100
        now_signal = _now_signal(p.get("action", "관망"), now_close, entry_close,
                                  p.get("stop_price"), tgt_price)
        row.update(status="채점완료", entry_close=round(entry_close, 1),
                   entry_date=entry_date, eval_date=str(ev_dt.date()),
                   eval_close=round(ev_close, 1), ret=round(ret, 2),
                   stop_hit=stop_hit,
                   target_up=round(target_up, 1) if target_up is not None else None,
                   target_hit=target_hit, verdict=verdict,
                   now_close=round(now_close, 1), now_ret=round(now_ret, 2),
                   now_signal=now_signal)
        rows.append(row)
    return rows


def _now_signal(action: str, now_close: float, entry_close: float,
                stop_price, target_price) -> str:
    """최신 종가 기준 '지금' 매도/보유 신호 (진입 포지션 관리용)."""
    if "진입" not in str(action):
        return "—"
    if target_price is not None and now_close >= float(target_price):
        return "🎯익절검토"
    if stop_price is not None and now_close <= float(stop_price):
        return "🔴손절"
    if stop_price is not None and now_close <= float(stop_price) * 1.02:
        return "🟠손절임박"
    if now_close <= entry_close * 0.97:
        return "🟠약세"
    return "🟢보유"


def _verdict(action: str, ret: float, stop_hit: bool) -> str:
    a = str(action)
    if "진입" in a:
        if stop_hit: return "❌손절"
        # 최종 평가 종가 기준으로만 판정 (손절선 장중 터치는 무시).
        # stop_hit은 now_signal/매도신호에서 참고용으로만 사용.
        if ret >= WIN_PCT:  return "✅성공"
        if ret <= -WIN_PCT: return "❌부진"
        return "➖중립"
    # 관망/회피: '안 사길 잘했나' 역채점
    if ret >= MISS_PCT:     return "⚠️놓침"
    if ret <= AVOID_OK_PCT: return "✅회피적중"
    return "➖중립"


def summary(scored_rows: list[dict]) -> dict:
    done = [r for r in scored_rows if r.get("status") == "채점완료"]
    ent = [r for r in done if "진입" in str(r.get("action"))]
    win = sum(1 for r in ent if r["verdict"] == "✅성공")
    loss = sum(1 for r in ent if r["verdict"] in ("❌손절", "❌부진"))
    avg_ret = (sum(r["ret"] for r in ent) / len(ent)) if ent else None
    return {"n_done": len(done), "n_entry": len(ent),
            "entry_win": win, "entry_loss": loss,
            "entry_win_rate": round(win / len(ent) * 100, 1) if ent else None,
            "entry_avg_ret": round(avg_ret, 2) if avg_ret is not None else None}
