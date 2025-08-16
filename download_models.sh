#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="models"
mkdir -p "$DEST_DIR"

URLS=(
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)

for URL in "${URLS[@]}"; do
  DEST_FILE="${DEST_DIR}/$(basename "$URL")"
  echo "Downloading to: $DEST_FILE"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --continue-at - -o "$DEST_FILE" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$DEST_FILE" "$URL"
  else
    echo "Error: please install curl or wget." >&2
    exit 1
  fi

  # Basic sanity check
  if [ ! -s "$DEST_FILE" ]; then
    echo "Download failed or file is empty: $DEST_FILE" >&2
    exit 1
  fi

  echo "Done: $DEST_FILE"
done

echo "All files downloaded successfully into: $DEST_DIR"
