"""Puts the service directory on sys.path for its own tests.

APPENDED, for the reason diffusionservice/tests/conftest.py gives: core/ and
this directory both contain a `main.py`, and core's tests import theirs as the
bare name `main`. Appending leaves core's first, so this dir only supplies
speech_params, which is the one thing these tests want.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
