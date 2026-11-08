"""The next release tag (.github/scripts/next_tag.sh, run by auto-tag.yaml).

A wrong tag is a real release, so every branch of the rule is pinned here:
the minor bump, the first release of a new major from .github/release-major,
and each mistake it must refuse.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "next_tag.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


MISSING = object()  # a file named but not there


def _next(latest, wanted, tmp_path):
    marker = tmp_path / "release-major"
    if wanted is not None and wanted is not MISSING:
        marker.write_text(f"{wanted}\n", encoding="utf-8")
    # Fed on stdin with LF endings: a Windows checkout gives the file CRLF,
    # which bash cannot run (CI checks out on Linux, so the workflow is fine).
    script = SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n")
    args = [latest] if wanted is None else [latest, marker.as_posix()]
    run = subprocess.run([BASH, "-s", *args], input=script, capture_output=True, text=True)
    return run.stdout.strip() if run.returncode == 0 else None


@pytest.mark.parametrize("latest, wanted, expected", [
    ("v2.101", "3", "v3.0"),     # merging v3: the first v3 release
    ("v3.0", "3", "v3.1"),       # then minor bumps as always
    ("v3.4", "3", "v3.5"),
    ("v2.101", None, "v2.102"),  # no file argument: as before
    ("v2.101", "3\r", "v3.0"),   # a Windows checkout's line ending
    ("", None, "v2.1"),          # no tag yet: as before
])
def test_the_next_tag(latest, wanted, expected, tmp_path):
    assert _next(latest, wanted, tmp_path) == expected


@pytest.mark.parametrize("latest, wanted", [
    ("v3.4", "2"),       # below the latest major: never tag backwards
    ("v2.101", "33"),    # a typo: never skip a major
    ("v2.101", "08"),    # bash would read it as octal
    ("v2.101", "x"),
    ("v2.101", ""),
    ("v2.x", "3"),       # a malformed latest tag
    ("v2.101", "2 3"),   # not one number
    ("v22.5", "2\n3"),   # two lines, never read as 23
    ("v2.101", MISSING),  # named but missing: never fall back to a minor bump
    ("v2.101", "18446744073709551619"),  # would wrap in bash arithmetic
])
def test_a_mistake_is_refused_not_tagged(latest, wanted, tmp_path):
    assert _next(latest, wanted, tmp_path) is None


def test_the_repo_file_names_this_major():
    marker = SCRIPT.parents[1] / "release-major"
    assert marker.read_text(encoding="utf-8").strip() == "3"


def test_the_release_gate_tests_on_the_python_the_image_runs():
    # release.yaml once ran its tests on 3.13 while tests.yaml and the image
    # used 3.14, so v3.0's release failed on code every PR check had passed.
    import re
    root = SCRIPT.parents[2]

    def versions(workflow):
        text = (root / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        return set(re.findall(r"python-version:\s*['\"]?([0-9.]+)", text))

    image = re.search(r"^FROM python:([0-9.]+)",
                      (root / "core" / "Dockerfile").read_text(encoding="utf-8"), re.M).group(1)
    assert versions("release.yaml") == versions("tests.yaml") == {image}
