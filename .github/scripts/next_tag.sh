#!/usr/bin/env bash
# Prints the next release tag (auto-tag.yaml; AGENTS.md -> Releasing):
#   next_tag.sh <latest tag, empty if none> [release-major file]
# The file holds the major version the next release belongs to. Greater than
# the latest tag's by one: the first release of a new major, v<major>.0.
# Equal (or no file argument): bump the minor as always. Anything else is a
# mistake (a typo would mint a real release), so it fails rather than guess:
# a file named but missing, a number that is not one, or one out of range.
# Numbers are at most 6 digits, so bash's integer arithmetic cannot wrap.
# Tested by core/tests/release_tag_tests.py.
set -euo pipefail

latest=${1:-}
file=${2:-}
if [ -z "$latest" ]; then
  latest="v2.0"
fi
if [[ ! "$latest" =~ ^v(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$ ]]; then
  echo "Refusing to increment malformed latest tag: $latest" >&2
  exit 1
fi

major=${latest#v}
minor=${major##*.}
major=${major%.*}

wanted=$major
if [ -n "$file" ]; then
  if [ ! -f "$file" ]; then
    echo "Refusing: $file is missing" >&2
    exit 1
  fi
  # $(...) drops the trailing newline; only CR goes besides (a Windows
  # checkout). Anything else, "2 3" or two lines, is malformed.
  wanted=$(tr -d '\r' < "$file")
  # No leading zeros: bash would read 08 as octal and fail the comparisons.
  if [[ ! "$wanted" =~ ^(0|[1-9][0-9]{0,5})$ ]]; then
    echo "Refusing a malformed $file: '$wanted'" >&2
    exit 1
  fi
fi

if (( wanted < major )); then
  echo "Refusing: $file ($wanted) is below the latest tag's major ($major)" >&2
  exit 1
elif (( wanted > major + 1 )); then
  echo "Refusing: $file ($wanted) skips past major $((major + 1)) from the latest tag ($latest)" >&2
  exit 1
elif (( wanted > major )); then
  next="v${wanted}.0"
else
  next="v${major}.$((minor + 1))"
fi
echo "$next"
