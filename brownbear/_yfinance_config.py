"""
Configure yfinance before any Yahoo API use.

yfinance 1.x stores cookies and timezone data in SQLite files under a shared
user cache directory. Multiple Jupyter kernels contend for those databases and
can block each other for minutes. Use a per-process cache directory instead.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from yfinance.cache import set_tz_cache_location

_configured = False


def _process_cache_dir():
    if override := os.environ.get('BROWNBEAR_YFINANCE_CACHE'):
        base = Path(override)
    else:
        base = Path(tempfile.gettempdir()) / 'brownbear-yfinance'
    return base / str(os.getpid())


def _close_yfinance_dbs():
    from yfinance.cache import _CookieDBManager, _ISINDBManager, _TzDBManager

    for manager in (_TzDBManager, _CookieDBManager, _ISINDBManager):
        try:
            manager.close_db()
        except Exception:
            pass


def _cleanup(cache_dir):
    _close_yfinance_dbs()
    shutil.rmtree(cache_dir, ignore_errors=True)


def configure():
    global _configured
    if _configured:
        return

    cache_dir = _process_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_tz_cache_location(str(cache_dir))
    atexit.register(_cleanup, cache_dir)
    _configured = True


configure()
