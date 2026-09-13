"""
screener/reporter.py
CSV + HTML 리포트 + 차트 이미지 저장
"""

import math
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TODAY = date.today().isoformat()

# Classification → 우선순위 / 색상 / 그룹명
CLS_META = {
    "Strong Momentum / Buyable":      {"order":1, "color":"#4ade80", "group":"신규매수 후보",  "bg":"rgba(74,222,128,0.08)"},
    "Breakout Candidate":             {"order":2, "color":"#34d399", "group":"신규매수 후보",  "bg":"rgba(52,211,153,0.08)"},
    "Pullback Candidate":             {"order":3, "color":"#60a5fa", "group":"눌림목 대기",    "bg":"rgba(96,165,250,0.08)"},
    "Strong Momentum / Extended":     {"order":4, "color":"#fbbf24", "group":"추격 위험",      "bg":"rgba(251,191,36,0.08)"},
    "Watch Only":                     {"order":5, "color":"#94a3b8", "group":"관심종목",       "bg":"rgba(148,163,184,0.05)"},
    "Avoid / Too Extended":           {"order":6, "color":"#f87171", "group":"제외",           "bg":"rgba(248,113,113,0.05)"},
    "Avoid / Weak Relative Strength": {"order":7, "color":"#64748b", "group":"제외",           "bg":"rgba(100,116,139,0.03)"},
}


def _fp(v, d=1, suffix="%") -> str:
    try:
        v = float(v)
    except Exception:
        return "-"
    if not math.isfinite(v):
        return "-"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{d}f}{suffix}"


def _pc(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "#94a3b8"
    if not math.isfinite(v):
        return "#94a3b8"
    return "#4ade80" if v >= 0 else "#f87171"


def _sc(score: float) -> str:
    if score >= 75:
        return "#f59e0b"
    elif score >= 60:
        return "#818cf8"
    return "#64748b"


def _bar(score: float, w: int = 70) -> str:
    sc = max(0, min(100, score if math.isfinite(score) else 50))
    c  = _sc(sc)
    fw = int(w * sc / 100)
    return (
        f'<div style="display:flex;align-items:center;gap:5px">'
        f'<div style="width:{w}px;height:5px;background:#1e293b;border-radius:3px;overflow:hidden">'
        f'<div style="width:{fw}px;height:100%;background:{c};border-radius:3px"></div></div>'
        f'<span style="font-size:11px;font-weight:700;color:{c}">{sc:.0f}</span>'
        f'</div>'
    )


# ════════════════════════════════════════════════════════════════════
# 차트 이미지 저장 (matplotlib)
# ════════════════════════════════════════════════════════════════════

def save_chart(ohlcv: pd.DataFrame, ticker: str, name: str = "") -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import warnings

        # 한글 폰트 경고 억제 — 제목에 한글 대신 티커만 사용
        warnings.filterwarnings("ignore", message="Glyph.*missing from font")

        close = ohlcv["Close"].tail(120)
        ma20  = close.rolling(20).mean()
        ma50  = close.rolling(50).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        facecolor="#0f172a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#64748b", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#1e293b")

        ax1.plot(close.index, close.values, color="#f1f5f9", linewidth=1.5, label="Close")
        ax1.plot(ma20.index, ma20.values,   color="#818cf8", linewidth=1,   label="MA20", linestyle="--")
        ax1.plot(ma50.index, ma50.values,   color="#f59e0b", linewidth=1,   label="MA50", linestyle="--")
        ax1.legend(fontsize=7, facecolor="#0f172a", labelcolor="#94a3b8", loc="upper left")
        # 제목에 티커만 사용 (한글 종목명 제외 → 폰트 경고 방지)
        ax1.set_title(ticker, color="#f1f5f9", fontsize=11, fontweight="bold", pad=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        if "Volume" in ohlcv.columns:
            vol = ohlcv["Volume"].tail(120)
            colors = ["#4ade80" if c >= o else "#f87171"
                      for c, o in zip(close.values, close.shift(1).values)]
            ax2.bar(vol.index, vol.values, color=colors, alpha=0.7, width=0.8)
            ax2.set_ylabel("Volume", color="#64748b", fontsize=8)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        plt.tight_layout()
        path = REPORT_DIR / f"chart_{ticker}_{TODAY}.png"
        plt.savefig(path, dpi=100, bbox_inches="tight", facecolor="#0f172a")
        plt.close(fig)
        return path
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"차트 저장 실패({ticker}): {e}")
        return None


# ════════════════════════════════════════════════════════════════════
# CSV 저장
# ════════════════════════════════════════════════════════════════════

def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = REPORT_DIR / f"{name}_{TODAY}.csv"
    df.to_csv(path, encoding="utf-8-sig")
    return path


# ════════════════════════════════════════════════════════════════════
# HTML 리포트
# ════════════════════════════════════════════════════════════════════

_HEAD = """<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8">
<title>모멘텀 스크리너 {today}</title>
<style>
  body{{background:#020817;color:#e2e8f0;font-family:'Noto Sans KR',sans-serif;padding:20px;margin:0}}
  h1{{font-size:20px;font-weight:900;color:#f1f5f9;margin-bottom:4px}}
  h2{{font-size:15px;font-weight:800;color:#f1f5f9;margin:24px 0 10px;border-left:3px solid {{color}};padding-left:10px}}
  .sub{{font-size:11px;color:#475569;margin-bottom:20px}}
  .meta{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
  .mc{{background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:10px 14px;text-align:center}}
  .mv{{font-size:18px;font-weight:800;color:#818cf8}}
  .ml{{font-size:10px;color:#475569;margin-top:2px}}
  table{{border-collapse:collapse;width:100%;font-size:11px;margin-bottom:30px}}
  th{{background:#0f172a;color:#64748b;font-weight:700;padding:7px 8px;text-align:left;
      border-bottom:1px solid rgba(255,255,255,0.08);white-space:nowrap}}
  td{{padding:7px 8px;border-bottom:1px solid rgba(255,255,255,0.04);vertical-align:middle}}
  tr:hover td{{background:rgba(255,255,255,0.02)}}
  .rank{{font-weight:800;width:24px;text-align:center}}
  .ticker{{font-family:monospace;font-weight:800;font-size:13px;color:#f1f5f9}}
  .flag{{font-size:9px;color:#fbbf24;font-weight:700}}
  .risk-flag{{font-size:9px;color:#f87171;font-weight:700}}
  a{{color:#60a5fa;text-decoration:none;font-size:10px}}
  a:hover{{text-decoration:underline}}
  .cls{{font-size:9px;padding:2px 6px;border-radius:4px;font-weight:700;white-space:nowrap}}
  img.chart{{width:200px;height:100px;object-fit:cover;border-radius:4px;border:1px solid rgba(255,255,255,0.08)}}
  .group-title{{font-size:14px;font-weight:800;color:#f1f5f9;margin:28px 0 8px;
    border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:6px}}
  .disclaimer{{font-size:10px;color:#1e293b;margin-top:30px;line-height:1.8}}
</style></head><body>
"""

_TAIL = """
<div class="disclaimer">
⚠️ 본 리포트는 정량 스크리닝 참고 자료이며 투자 추천이 아닙니다.
최종 매수 판단은 차트·뉴스·공시·실적을 직접 확인한 후 본인 책임하에 이루어져야 합니다.
</div>
</body></html>"""


def _cls_badge(cls: str) -> str:
    meta = CLS_META.get(cls, {"color": "#64748b"})
    c    = meta["color"]
    return (
        f'<span class="cls" style="background:{c}22;color:{c};border:1px solid {c}44">'
        f'{cls}</span>'
    )


def _score_row_us(rank: int, ticker: str, row: pd.Series, show_chart: bool) -> str:
    cls   = row.get("classification", "")
    meta  = CLS_META.get(cls, {"color":"#64748b","bg":"rgba(255,255,255,0.02)"})
    rc    = "#f59e0b" if rank == 1 else "#818cf8" if rank <= 3 else "#475569"
    flag  = row.get("risk_flag", "")
    chart_path = REPORT_DIR / f"chart_{ticker}_{TODAY}.png"
    chart_html = (
        f'<img class="chart" src="{chart_path.name}">'
        if show_chart and chart_path.exists() else ""
    )

    links = (
        f'<a href="{row.get("yf_link","#")}" target="_blank">YF</a> '
        f'<a href="{row.get("news_link","#")}" target="_blank">News</a> '
        f'<a href="{row.get("edgar_link","#")}" target="_blank">Edgar</a>'
    )

    return (
        f'<tr style="background:{meta["bg"]}">'
        f'<td class="rank" style="color:{rc}">{"👑" if rank==1 else f"#{rank}"}</td>'
        f'<td><span class="ticker">{ticker}</span>'
        f'{"<br><span class=risk-flag>⚠ " + flag + "</span>" if flag else ""}</td>'
        f'<td style="font-size:10px;color:#94a3b8">{str(row.get("company",""))[:18]}</td>'
        f'<td style="font-size:10px;color:#64748b">{str(row.get("sector",""))[:14]}</td>'
        f'<td style="color:{_pc(row.get("ret_5d"))};font-weight:700">{_fp(row.get("ret_5d"))}</td>'
        f'<td style="color:{_pc(row.get("ret_20d"))};font-weight:700">{_fp(row.get("ret_20d"))}</td>'
        f'<td style="color:{_pc(row.get("ret_60d"))};font-weight:700">{_fp(row.get("ret_60d"))}</td>'
        f'<td style="color:{_pc(row.get("rs_spy_20"))};font-weight:700">{_fp(row.get("rs_spy_20"),suffix="pp")}</td>'
        f'<td style="color:{_pc(row.get("rs_qqq_20"))};font-weight:700">{_fp(row.get("rs_qqq_20"),suffix="pp")}</td>'
        f'<td style="color:{_pc(row.get("rs_sector_20"))};font-weight:700">{_fp(row.get("rs_sector_20"),suffix="pp")}</td>'
        f'<td style="color:{"#4ade80" if row.get("week52_prox",0)>=90 else "#fbbf24" if row.get("week52_prox",0)>=75 else "#94a3b8"}'
        f';font-weight:700">{_fp(row.get("week52_prox"),0,"%")}</td>'
        f'<td style="color:#94a3b8">{_fp(row.get("vol_surge"),2,"x")}</td>'
        f'<td style="color:{"#4ade80" if row.get("above_20dma") else "#f87171"}">{"✓" if row.get("above_20dma") else "✗"}</td>'
        f'<td style="color:{"#4ade80" if row.get("above_50dma") else "#f87171"}">{"✓" if row.get("above_50dma") else "✗"}</td>'
        f'<td style="color:{_pc(row.get("dist_20dma"))}">{_fp(row.get("dist_20dma"))}</td>'
        f'<td>{_bar(row.get("momentum_score",0))}</td>'
        f'<td>{_bar(row.get("volume_score",0))}</td>'
        f'<td>{_bar(row.get("buyable_score",0))}</td>'
        f'<td><span style="color:#f87171;font-weight:700;font-size:11px">{row.get("top_risk_score",0):.0f}</span></td>'
        f'<td><span style="font-size:13px;font-weight:800;color:{_sc(row.get("final_score",0))}">'
        f'{row.get("final_score",0):.1f}</span></td>'
        f'<td>{_cls_badge(cls)}</td>'
        f'<td style="font-size:10px;color:#64748b">{row.get("earnings_date","")}</td>'
        f'<td>{links}</td>'
        f'<td>{chart_html}</td>'
        f'</tr>'
    )


def _score_row_kr(rank: int, ticker: str, row: pd.Series, show_chart: bool) -> str:
    cls  = row.get("classification", "")
    meta = CLS_META.get(cls, {"color":"#64748b","bg":"rgba(255,255,255,0.02)"})
    rc   = "#f59e0b" if rank == 1 else "#818cf8" if rank <= 3 else "#475569"
    flag = row.get("risk_flag", "")
    chart_path = REPORT_DIR / f"chart_{ticker}_{TODAY}.png"
    chart_html = (
        f'<img class="chart" src="{chart_path.name}">'
        if show_chart and chart_path.exists() else ""
    )

    links = (
        f'<a href="{row.get("naver_link","#")}" target="_blank">네이버</a> '
        f'<a href="{row.get("news_link","#")}" target="_blank">뉴스</a> '
        f'<a href="{row.get("dart_link","#")}" target="_blank">DART</a> '
        f'<a href="{row.get("kind_link","#")}" target="_blank">KIND</a>'
    )

    def amt(v):
        try:
            v = float(v)
        except Exception:
            return "-"
        if not math.isfinite(v):
            return "-"
        c = "#4ade80" if v >= 0 else "#f87171"
        return f'<span style="color:{c};font-weight:700">{v/10000:+.0f}만주</span>'

    return (
        f'<tr style="background:{meta["bg"]}">'
        f'<td class="rank" style="color:{rc}">{"👑" if rank==1 else f"#{rank}"}</td>'
        f'<td><span class="ticker">{ticker}</span>'
        f'{"<br><span class=risk-flag>⚠ " + flag + "</span>" if flag else ""}</td>'
        f'<td style="font-size:11px;color:#f1f5f9">{row.get("name","")}</td>'
        f'<td><span style="font-size:9px;padding:1px 5px;border-radius:3px;font-weight:700;'
        f'background:{"rgba(239,68,68,0.2)" if row.get("market")=="KOSDAQ" else "rgba(59,130,246,0.2)"};'
        f'color:{"#fca5a5" if row.get("market")=="KOSDAQ" else "#93c5fd"}">'
        f'{row.get("market","")}</span> '
        f'<span style="font-size:9px;color:#64748b">{str(row.get("sector",""))[:10]}</span></td>'
        f'<td style="color:{_pc(row.get("ret_5d"))};font-weight:700">{_fp(row.get("ret_5d"))}</td>'
        f'<td style="color:{_pc(row.get("ret_20d"))};font-weight:700">{_fp(row.get("ret_20d"))}</td>'
        f'<td style="color:{_pc(row.get("ret_60d"))};font-weight:700">{_fp(row.get("ret_60d"))}</td>'
        f'<td style="color:{_pc(row.get("rs_mkt_20"))};font-weight:700">{_fp(row.get("rs_mkt_20"),suffix="pp")}</td>'
        f'<td style="color:{_pc(row.get("rs_sector_20"))};font-weight:700">{_fp(row.get("rs_sector_20"),suffix="pp")}</td>'
        f'<td style="color:#94a3b8">{_fp(row.get("vol_surge"),2,"x")}</td>'
        f'<td>{amt(row.get("foreign_5d"))}</td>'
        f'<td>{amt(row.get("inst_5d"))}</td>'
        f'<td style="color:{"#4ade80" if row.get("week52_prox",0)>=90 else "#fbbf24" if row.get("week52_prox",0)>=75 else "#94a3b8"}'
        f';font-weight:700">{_fp(row.get("week52_prox"),0,"%")}</td>'
        f'<td style="color:{"#4ade80" if row.get("above_20dma") else "#f87171"}">{"✓" if row.get("above_20dma") else "✗"}</td>'
        f'<td style="color:{"#4ade80" if row.get("above_60dma") else "#f87171"}">{"✓" if row.get("above_60dma") else "✗"}</td>'
        f'<td style="color:{_pc(row.get("dist_20dma"))}">{_fp(row.get("dist_20dma"))}</td>'
        f'<td>{_bar(row.get("momentum_score",0))}</td>'
        f'<td>{_bar(row.get("volume_score",0))}</td>'
        f'<td>{_bar(row.get("buyable_score",0))}</td>'
        f'<td><span style="color:#f87171;font-weight:700;font-size:11px">{row.get("top_risk_score",0):.0f}</span></td>'
        f'<td><span style="font-size:13px;font-weight:800;color:{_sc(row.get("final_score",0))}">'
        f'{row.get("final_score",0):.1f}</span></td>'
        f'<td>{_cls_badge(cls)}</td>'
        f'<td>{links}</td>'
        f'<td>{chart_html}</td>'
        f'</tr>'
    )


def _table_us(top: pd.DataFrame, show_chart: bool) -> str:
    if top is None or top.empty:
        return "<p style='color:#475569'>미국 데이터 없음</p>"

    # Classification 순으로 그룹화
    groups_html = ""
    ordered_cls = sorted(CLS_META.keys(), key=lambda x: CLS_META[x]["order"])
    rank_map = {t: i+1 for i, t in enumerate(top.index)}

    for cls in ordered_cls:
        subset = top[top["classification"] == cls]
        if subset.empty:
            continue
        meta = CLS_META[cls]
        rows_html = "".join(
            _score_row_us(rank_map[t], t, row, show_chart)
            for t, row in subset.iterrows()
        )
        groups_html += (
            f'<div class="group-title" style="color:{meta["color"]}">'
            f'{meta["group"]} — {cls} ({len(subset)}개)</div>'
            f'<table>'
            f'<thead><tr>'
            f'<th>#</th><th>티커</th><th>회사</th><th>섹터</th>'
            f'<th>5일</th><th>20일</th><th>60일</th>'
            f'<th>RS/SPY</th><th>RS/QQQ</th><th>RS/섹터</th>'
            f'<th>52주%</th><th>거래량×</th><th>20MA</th><th>50MA</th><th>MA이격</th>'
            f'<th>Momentum</th><th>Volume</th><th>Buyable</th>'
            f'<th style="color:#f87171">Risk</th><th>Final</th>'
            f'<th>분류</th><th>실적일</th><th>링크</th><th>차트</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
        )

    return groups_html


def _table_kr(top: pd.DataFrame, show_chart: bool) -> str:
    if top is None or top.empty:
        return "<p style='color:#475569'>한국 데이터 없음</p>"

    groups_html = ""
    ordered_cls = sorted(CLS_META.keys(), key=lambda x: CLS_META[x]["order"])
    rank_map = {t: i+1 for i, t in enumerate(top.index)}

    for cls in ordered_cls:
        subset = top[top["classification"] == cls]
        if subset.empty:
            continue
        meta = CLS_META[cls]
        rows_html = "".join(
            _score_row_kr(rank_map[t], t, row, show_chart)
            for t, row in subset.iterrows()
        )
        groups_html += (
            f'<div class="group-title" style="color:{meta["color"]}">'
            f'{meta["group"]} — {cls} ({len(subset)}개)</div>'
            f'<table>'
            f'<thead><tr>'
            f'<th>#</th><th>코드</th><th>종목명</th><th>시장/업종</th>'
            f'<th>5일</th><th>20일</th><th>60일</th>'
            f'<th>시장RS</th><th>업종RS</th><th>거래량×</th>'
            f'<th>외국인5일</th><th>기관5일</th>'
            f'<th>52주%</th><th>20MA</th><th>60MA</th><th>MA이격</th>'
            f'<th>Momentum</th><th>Volume</th><th>Buyable</th>'
            f'<th style="color:#f87171">Risk</th><th>Final</th>'
            f'<th>분류</th><th>링크</th><th>차트</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
        )

    return groups_html


def save_html_report(
    us_top:    Optional[pd.DataFrame],
    kr_top:    Optional[pd.DataFrame],
    failed_us: list,
    failed_kr: list,
    show_chart: bool = True,
) -> Path:
    us_n = len(us_top) if us_top is not None else 0
    kr_n = len(kr_top) if kr_top is not None else 0

    # 그룹별 분포 요약
    def group_summary(top: Optional[pd.DataFrame]) -> str:
        if top is None or top.empty:
            return ""
        cnts = top["classification"].value_counts()
        parts = []
        for cls, cnt in cnts.items():
            meta = CLS_META.get(cls, {"color": "#64748b"})
            parts.append(
                f'<span style="color:{meta["color"]};font-weight:700">{cls}</span>'
                f'<span style="color:#475569"> {cnt}개</span>'
            )
        return " &nbsp;|&nbsp; ".join(parts)

    html = (
        _HEAD.format(today=TODAY)
        + f'<h1>📈 모멘텀 스크리너 — {TODAY}</h1>'
        + f'<div class="sub">정량 스크리닝 참고 자료 · 투자 추천 아님 · '
        + f'수집 실패 US {len(failed_us)}개 / KR {len(failed_kr)}개</div>'
        + f'<div class="meta">'
        + f'<div class="mc"><div class="mv">{us_n}</div><div class="ml">미국 Top</div></div>'
        + f'<div class="mc"><div class="mv">{kr_n}</div><div class="ml">한국 Top</div></div>'
        + f'</div>'
        + f'<h2 style="border-color:#60a5fa;color:#60a5fa">🇺🇸 미국 Top {us_n}</h2>'
        + f'<div style="font-size:11px;color:#475569;margin-bottom:10px">{group_summary(us_top)}</div>'
        + _table_us(us_top, show_chart)
        + f'<h2 style="border-color:#f87171;color:#f87171">🇰🇷 한국 Top {kr_n}</h2>'
        + f'<div style="font-size:11px;color:#475569;margin-bottom:10px">{group_summary(kr_top)}</div>'
        + _table_kr(kr_top, show_chart)
        + _TAIL
    )

    path = REPORT_DIR / f"momentum_report_{TODAY}.html"
    path.write_text(html, encoding="utf-8")
    return path
