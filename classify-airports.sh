#!/usr/bin/env bash
# classify-airports.sh — on-demand FAA airport classifier.
#
# Prompts for a destination folder and one or more FAA LID codes, then for
# EACH code:
#   1. Downloads the latest FAA data (airport-diagram PDF, approach chart,
#      NASR airspace class + runway rows) via fetch_airport.py into a
#      de-duplicated folder under the destination (ord, ord-1, ord-2, ...).
#   2. Runs classify.sh on the freshly-downloaded PDF.
#
# Re-running on an already-classified airport is safe: the folder is
# de-duplicated, never overwritten. The "Place approach lights?" prompt is
# forwarded to classify.sh via PLACE_LIGHTS. Fresh per-airport NASR runway
# data is forwarded via NASR_RWY_CSV.
#
# The destination default comes from classify-wrapper.conf (DEFAULT_DEST_DIR);
# press Enter to accept it or type another path for this run only.

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT/.venv/bin/python3"
CONFIG="$PROJECT/classify-wrapper.conf"

# Hardcoded fallback default. Overridden by classify-wrapper.conf if present.
DEFAULT_DEST_DIR="/Users/lukehogan/Documents/startups/aoa/products/artwork/airports"

if [[ -f "$CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG"
fi

DEST_DIR="$DEFAULT_DEST_DIR"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
read -e -p "Destination folder [$DEST_DIR]: " input
if [[ -n "${input:-}" ]]; then
    DEST_DIR="$input"
fi

read -e -p "Airport LID code(s), comma-separated: " codes_input
if [[ -z "${codes_input:-}" ]]; then
    echo "ERROR: no airport codes provided" >&2
    exit 1
fi

read -e -p "Place approach lights? [y/N] " place_lights_in
case "$(echo "$place_lights_in" | tr '[:upper:]' '[:lower:]')" in
    y|yes) export PLACE_LIGHTS=1 ;;
    *)     export PLACE_LIGHTS=0 ;;
esac
echo

# ---------------------------------------------------------------------------
# Normalize codes: split on comma, trim whitespace, lowercase.
# ---------------------------------------------------------------------------
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

mkdir -p "$DEST_DIR"

echo "Destination: $DEST_DIR"
echo "Codes:       ${CODES[*]}"
echo

# ---------------------------------------------------------------------------
# Process each code: fetch latest FAA data, then classify.
# ---------------------------------------------------------------------------
failed=0
for code in "${CODES[@]}"; do
    echo "▶  $code: fetch latest FAA data"
    # Capture fetch_airport.py stdout so we can read FOLDER= / PDF=.
    if ! fetch_out="$("$PYTHON" "$PROJECT/python/fetch_airport.py" \
            --lid "$code" --dest-parent "$DEST_DIR")"; then
        echo "   FAILED: could not fetch $code (no diagram or download error)" >&2
        (( failed++ )) || true
        continue
    fi

    folder="$(echo "$fetch_out" | sed -n 's/^FOLDER=//p')"
    pdf="$(echo "$fetch_out" | sed -n 's/^PDF=//p')"
    if [[ -z "$folder" || -z "$pdf" || ! -f "$pdf" ]]; then
        echo "   FAILED: fetch produced no usable PDF for $code" >&2
        (( failed++ )) || true
        continue
    fi

    # Forward fresh per-airport NASR runway data to classify.sh if written.
    export NASR_RWY_CSV="$folder/$code-nasr-rwy.csv"

    echo "▶  $code: classify ($folder)"
    if ! bash "$PROJECT/classify.sh" "$pdf"; then
        echo "   FAILED: classify $code" >&2
        (( failed++ )) || true
    fi
    echo
done

if [[ $failed -eq 0 ]]; then
    echo "Done. Processed: ${CODES[*]}"
else
    echo "$failed airport(s) failed."
    exit 1
fi
