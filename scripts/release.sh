#!/usr/bin/env bash
# Check the CHANGELOG, lint and test, then build the app folder (scripts/package.sh) for the
# manual upload to the shared drive. Arguments go to scripts/package.sh: [--wheel] [folder].
set -euo pipefail

cd "$(dirname "$0")/.."

version="$(poetry version --short)"
# release-please writes "## [1.2.0](...)"; the first entry is "## 1.0.0 (...)"
if ! grep -qE "^## \[?${version//./\\.}\]?[ (]" CHANGELOG.md; then
  echo "CHANGELOG.md has no entry for ${version}." >&2
  exit 1
fi

mise run lint
.venv/bin/pytest -q
scripts/package.sh "$@"
echo "Upload the contents of release/ (or the zip) to the app folder on the shared drive."
