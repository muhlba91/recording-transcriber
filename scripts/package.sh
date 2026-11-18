#!/usr/bin/env bash
# Build an upload-ready app folder in release/ and a zip of it in dist/.
# Usage: scripts/package.sh [--wheel] [folder]
#   folder   an example folder with config.yaml (anywhere); default examples/generic
#   --wheel  build the local wheel and add it (release candidates, or until PyPI serves the version)
# Used by `mise run package`, scripts/release.sh and the Build job.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"

with_wheel=false
folder=""
for argument in "$@"; do
  case "${argument}" in
    --wheel) with_wheel=true ;;
    -*)
      echo "Unknown option ${argument}. Usage: scripts/package.sh [--wheel] [folder]" >&2
      exit 1
      ;;
    *) folder="${argument}" ;;
  esac
done
# a relative folder is relative to the caller's directory, the default to the repository
folder="${folder:-${root}/examples/generic}"

if [[ ! -f "${folder}/config.yaml" ]]; then
  echo "${folder} has no config.yaml." >&2
  exit 1
fi
folder="$(cd "${folder}" && pwd)"
name="$(basename "${folder}")"
cd "${root}"

version="$(poetry version --short)"

rm -rf release
mkdir release
if [[ "${with_wheel}" == true ]]; then
  poetry build --quiet --format wheel
  cp "dist/recording_transcriber-${version}-py3-none-any.whl" release/
fi
cp "${folder}/config.yaml" release/
if [[ -f "${folder}/glossary_global.txt" ]]; then
  cp "${folder}/glossary_global.txt" release/
fi
if [[ -d "${folder}/prompts" ]]; then
  cp -R "${folder}/prompts" release/prompts
fi
cp colab/recording-transcriber.ipynb how-to.md CHANGELOG.md release/

# the zip holds the folder as audio_transcription/; it stays out of release/, the folder itself
staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT
cp -R release "${staging}/audio_transcription"
mkdir -p dist
zip_file="dist/recording-transcriber-${version}-${name}.zip"
rm -f "${zip_file}"
(cd "${staging}" && zip -qr "${OLDPWD}/${zip_file}" audio_transcription)

echo "release/ holds ${name} for version ${version}:"
ls -1 release
echo "app folder zip: ${zip_file}"
