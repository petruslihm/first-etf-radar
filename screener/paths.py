"""
캐시 경로 + 데이터 기준일(스탬프) 중앙 관리.

목적:
  1) 캐시를 zip 압축 해제 위치 '바깥'에 두어, 코드를 새 zip으로 갈아끼워도
     그날 수집한 데이터가 사라지지 않게 한다.
  2) 수집한 '데이터 기준일'을 스탬프 파일에 기록해, 앱에서 표시하고
     파생 캐시(점수/섹터)를 그 날짜에 묶는다.

저장 위치 우선순위:
  - 환경변수 ETF_DATA_DIR 가 있으면 그 경로
  - 없으면 ~/.etf-radar-cache  (사용자 홈 — zip 바깥이라 보존됨)
"""
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional


def _resolve_data_dir() -> Path:
    env = os.environ.get("ETF_DATA_DIR")
    base = Path(env).expanduser() if env else (Path.home() / ".etf-radar-cache")
    base.mkdir(parents=True, exist_ok=True)
    return base


DATA_DIR   = _resolve_data_dir()
CACHE_DIR  = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_STAMP = CACHE_DIR / "data_stamp.txt"
TODAY  = date.today().isoformat()


def get_data_date() -> Optional[str]:
    """현재 캐시된 데이터의 기준일(ISO). 없으면 None."""
    try:
        if _STAMP.exists():
            s = _STAMP.read_text().strip()
            return s or None
    except Exception:
        pass
    return None


def set_data_date(d: Optional[str] = None, only_if_absent: bool = False):
    """데이터 기준일 기록. d 미지정 시 오늘. only_if_absent면 없을 때만 기록."""
    if only_if_absent and get_data_date():
        return
    try:
        _STAMP.write_text(d or TODAY)
    except Exception:
        pass


def clear_data_date():
    _STAMP.unlink(missing_ok=True)


def data_age_days() -> Optional[int]:
    d = get_data_date()
    if not d:
        return None
    try:
        return (date.today() - datetime.fromisoformat(d).date()).days
    except Exception:
        return None


def data_date_key() -> str:
    """파생 캐시(factors/sector)에 쓸 날짜 키. 스탬프 우선, 없으면 오늘."""
    return get_data_date() or TODAY
