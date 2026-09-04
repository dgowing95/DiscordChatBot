"""Puts the service directory on sys.path for its own tests.

APPEND, not insert: both core/ and diffusionservice/ contain a `main.py`, and
core/tests/*_tests.py import theirs as the bare name `main`. Prepending (which
is what pyproject.toml's `pythonpath` does) would shadow core's with the
service's, and every test that imports main would then be exercising a FastAPI
app that needs torch. Appending leaves the `core` entry ahead of it, so the
service dir only ever supplies names core does not already have --
generation_params, which is the one thing these tests want.

This also keeps `pythonpath` in pyproject.toml as `core` only, so nothing about
the core suite's import resolution changes.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
