#!/usr/bin/env bash
# Fetch the TNDM release (Heidari et al., arXiv:2610.24721) at the commit the paper used.
# It provides the coalition cubes, the source's saved results and its fitters. It is not
# redistributed in this repository; its terms are its authors'.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/TNDM-data-attribution/TNDM
COMMIT=b20e0ebf5f3f9ff9616ea6af2bd98651ad291614
if [ ! -d third_party/TNDM/.git ]; then
  mkdir -p third_party
  git clone "$URL" third_party/TNDM
fi
git -C third_party/TNDM checkout --quiet "$COMMIT"
echo "third_party/TNDM at $(git -C third_party/TNDM rev-parse --short HEAD)"
