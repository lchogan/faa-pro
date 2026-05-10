#!/usr/bin/env bash
# classify-airports.sh — interactive wrapper around classify.sh that
# moves airport folders out of the daily faa-downloader dump into the
# long-term artwork folder, then runs classify.sh on each one.
#
# Workflow:
#   1. Prompt for the source folder (where today's downloads live).
#   2. Prompt for the destination folder (long-term artwork home).
#   3. Prompt for one or more comma-separated airport codes.
#   4. For each code, `mv` the per-code subfolder from source to
#      destination, then invoke classify.sh on the PDF in its new
#      location.
#
# Defaults for the two folders come from `classify-wrapper.conf` next
# to this script. If that file is missing or doesn't define a key,
# the hardcoded fallbacks below are used. Pressing Enter at a prompt
# accepts the current default; typing a new path overrides it for
# this run only (the config file is NOT modified — edit it by hand
# to change the persistent default).

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$PROJECT/classify-wrapper.conf"

# Hardcoded fallback defaults. These are overridden by the config
# file if it exists. Edit classify-wrapper.conf to change persistent
# defaults; do NOT edit these.
DEFAULT_SOURCE_DIR="/Users/lukehogan/AOA-Code/faa-downloader/airports-2026-05-06"
DEFAULT_DEST_DIR="/Users/lukehogan/Documents/startups/aoa/products/artwork/airports"

if [[ -f "$CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG"
fi

# Effective values for this run (start at config-or-fallback default;
# user can override at the prompt).
SOURCE_DIR="$DEFAULT_SOURCE_DIR"
DEST_DIR="$DEFAULT_DEST_DIR"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# `read -e -p` gives readline editing on the prompt; pressing Enter
# returns an empty string, in which case we keep the current default.
read -e -p "Source folder [$SOURCE_DIR]: " input
if [[ -n "${input:-}" ]]; then
    SOURCE_DIR="$input"
fi

read -e -p "Destination folder [$DEST_DIR]: " input
if [[ -n "${input:-}" ]]; then
    DEST_DIR="$input"
fi

read -e -p "Airport code(s), comma-separated: " codes_input
if [[ -z "${codes_input:-}" ]]; then
    echo "ERROR: no airport codes provided" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Validate folders
# ---------------------------------------------------------------------------

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "ERROR: source folder does not exist: $SOURCE_DIR" >&2
    exit 1
fi

# Create destination if missing — the long-term artwork folder is
# expected to exist already, but if it's a fresh install or a typo,
# making it explicit beats failing later inside `mv`.
mkdir -p "$DEST_DIR"

# ---------------------------------------------------------------------------
# Process each airport code
# ---------------------------------------------------------------------------

# Split comma-separated input into an array, trim whitespace, lowercase.
IFS=',' read -ra RAW_CODES <<< "$codes_input"
CODES=()
for raw in "${RAW_CODES[@]}"; do
    trimmed="$(echo "$raw" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    if [[ -n "$trimmed" ]]; then
        CODES+=("$trimmed")
    fi
done

if [[ ${#CODES[@]} -eq 0 ]]; then
    echo "ERROR: no valid airport codes after trimming" >&2
    exit 1
fi

echo
echo "Source:      $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo "Codes:       ${CODES[*]}"
echo
echo "About to MOVE each code's subfolder from source to destination,"
echo "then run classify.sh on it in the new location."
read -e -p "Proceed? [y/N] " confirm
confirm_lc="$(echo "$confirm" | tr '[:upper:]' '[:lower:]')"
case "$confirm_lc" in
    y|yes) ;;
    *) echo "Aborted."; exit 0 ;;
esac
echo

for code in "${CODES[@]}"; do
    src_path="$SOURCE_DIR/$code"
    dst_path="$DEST_DIR/$code"
    pdf="$dst_path/$code-faa.pdf"

    if [[ ! -d "$src_path" ]]; then
        echo "WARN: $src_path does not exist, skipping $code" >&2
        continue
    fi
    if [[ -e "$dst_path" ]]; then
        # Refuse to overwrite. The user has to resolve this manually
        # (rename, delete, or merge). Silent overwrite would risk
        # losing prior labeling work in the destination folder.
        echo "ERROR: $dst_path already exists — refusing to overwrite" >&2
        exit 1
    fi

    echo "▶  $code: mv $src_path -> $dst_path"
    mv "$src_path" "$dst_path"

    if [[ ! -f "$pdf" ]]; then
        echo "ERROR: expected PDF at $pdf, none found after move" >&2
        exit 1
    fi

    echo "▶  $code: classify"
    bash "$PROJECT/classify.sh" "$pdf"
    echo
done

echo "Done. Processed: ${CODES[*]}"
