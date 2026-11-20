#!/usr/bin/env python3
"""Run the LIBERO evaluation client with video encoding disabled."""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any

import tyro


CLIENT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CLIENT_DIR))

import eval_client as _client  # noqa: E402


def _discard_video(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


_client.imageio.mimwrite = _discard_video


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(_client.eval_libero)
