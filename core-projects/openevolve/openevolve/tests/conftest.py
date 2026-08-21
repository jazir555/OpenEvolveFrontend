"""
Shared pytest configuration for OpenEvolve integration tests.

The LoongFlow adapter logs user-facing status banners that contain Unicode
box-drawing characters. On Windows consoles (cp1252) the stdlib
``logging.lastResort`` handler raises ``UnicodeEncodeError`` when it tries to
write those messages, which can mark unrelated tests as failed. We swap the
last-resort handler for one that tolerates encoding errors so tests stay
deterministic regardless of the active console encoding.
"""

import logging
import sys


def _install_safe_last_resort() -> None:
    try:
        stream = open(
            sys.__stderr__.fileno(),
            "w",
            encoding="utf-8",
            errors="replace",
            closefd=False,
        )
    except Exception:
        stream = sys.__stderr__
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logging.lastResort = handler


def pytest_configure(config):
    _install_safe_last_resort()
