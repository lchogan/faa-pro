#!/usr/bin/env bash
# export_paths.sh — Phase 1 of the character-recognition build plan.
#
# Walks the FAA-downloader corpus directories and, for each <code>-faa.pdf,
# runs ClassifyAirport.jsx in export-only mode to produce:
#   data/char_corpus/<code>/<code>_paths.csv
#   data/char_corpus/<code>/<code>_paths_edges.csv
#
# These CSVs feed Phase 2 (python/build_char_training.py), which aligns each
# polygon to the nearest PDF text-token character to build distant-supervision
# training data for the 36-class character classifier.
#
# Resume-friendly: airports whose CSV already exists are skipped, so re-running
# after an interrupt picks up where it left off.
#
# Usage:
#   bash export_paths.sh                         # process all default roots
#   bash export_paths.sh path/to/some/airport    # process a single folder
#   bash export_paths.sh --root /custom/dir      # add an extra root
#
# Default roots: faa-downloader/airports + faa-downloader/airports-dup
# (airports-class is intentionally excluded — geometry/text integrity uncertain.)
#
# Requires:
#   - Adobe Illustrator installed and accessible via AppleScript
#   - ClassifyAirport.jsx in this directory (with export_only support)

set -euo pipefail

# Script lives in faa-pro/python/char_training_legacy/. PROJECT_ROOT is
# the faa-pro project root (two levels up).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLASSIFY_JSX="$SCRIPT_DIR/ClassifyAirport.jsx"
CORPUS_DIR="$PROJECT_ROOT/data/char_training_legacy/char_corpus"
CONFIG="/tmp/classify_config.json"

DEFAULT_ROOTS=(
    "/Users/lukehogan/AOA-Code/faa-downloader/airports"
    "/Users/lukehogan/AOA-Code/faa-downloader/airports-dup"
)

# ---------------------------------------------------------------------------
# run_jsx <path-to-jsx>
# Tells Illustrator (via AppleScript) to run the given JSX file and waits
# for it to finish before returning. Reads file content directly to avoid
# #include reliability issues in Illustrator 2022+.
# ---------------------------------------------------------------------------
run_jsx() {
    local jsx="$1"
    osascript - "$jsx" <<'APPLESCRIPT' > /dev/null
on run argv
    set jsx_path to item 1 of argv
    tell application "Adobe Illustrator"
        activate
        set scriptText to read (POSIX file jsx_path) as «class utf8»
        do javascript scriptText
    end tell
end run
APPLESCRIPT
}

# ---------------------------------------------------------------------------
# process_one <pdf-path>
# Resolves paths, writes the JSX config, runs ClassifyAirport.jsx in
# export-only mode, and verifies the CSV landed.
# ---------------------------------------------------------------------------
process_one() {
    local pdf
    pdf="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

    local airport
    airport="$(basename "$pdf" | sed 's/-faa\.pdf$//' | tr '[:upper:]' '[:lower:]')"

    local out_folder="$CORPUS_DIR/$airport"
    local paths_csv="$out_folder/${airport}_paths.csv"
    local edges_csv="$out_folder/${airport}_paths_edges.csv"

    if [[ -f "$paths_csv" && -f "$edges_csv" ]]; then
        echo "▶  $airport [skip — CSVs already exist]"
        return 0
    fi

    mkdir -p "$out_folder"

    echo "▶  $airport"

    # The JSX reads paths_csv / edges_csv directly from the config (new keys
    # added alongside export_only). Other fields kept for parity with
    # classify.sh in case the JSX ever consults them.
    cat > "$CONFIG" <<JSON
{
  "pdf_path":     "$pdf",
  "airport":      "$airport",
  "folder":       "$(dirname "$pdf")",
  "paths_csv":    "$paths_csv",
  "edges_csv":    "$edges_csv",
  "json_path":    "$out_folder/${airport}_predictions.json",
  "ready_ai":     "$out_folder/${airport}-ready.ai",
  "diagram_ai":   "$out_folder/${airport}-diagram.ai",
  "export_only":  true
}
JSON

    run_jsx "$CLASSIFY_JSX"

    if [[ ! -f "$paths_csv" ]]; then
        echo "   ERROR: no CSV produced — check Illustrator log /tmp/classify_${airport}.log"
        rm -f "$CONFIG"
        return 1
    fi

    local path_count
    path_count=$(( $(wc -l < "$paths_csv") - 1 ))
    echo "   ✓ exported $path_count paths"

    rm -f "$CONFIG"
}

# ---------------------------------------------------------------------------
# walk_root <root-dir>
# Finds every <code>/<code>-faa.pdf under root-dir (one level deep).
# ---------------------------------------------------------------------------
walk_root() {
    local root="$1"
    if [[ ! -d "$root" ]]; then
        echo "   WARN: root does not exist: $root" >&2
        return
    fi
    find "$root" -mindepth 2 -maxdepth 2 -name "*-faa.pdf" -type f | sort
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
mkdir -p "$CORPUS_DIR"

declare -a ROOTS
declare -a EXPLICIT_PDFS

if [[ $# -eq 0 ]]; then
    ROOTS=("${DEFAULT_ROOTS[@]}")
else
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --root)
                ROOTS+=("$2")
                shift 2
                ;;
            *)
                EXPLICIT_PDFS+=("$1")
                shift
                ;;
        esac
    done
    # If no explicit --root and no PDFs given, fall back to defaults.
    if [[ ${#ROOTS[@]} -eq 0 && ${#EXPLICIT_PDFS[@]} -eq 0 ]]; then
        ROOTS=("${DEFAULT_ROOTS[@]}")
    fi
fi

# Collect work list. (Avoid expanding empty arrays under `set -u`.)
declare -a PDFS
if [[ ${#ROOTS[@]} -gt 0 ]]; then
    for r in "${ROOTS[@]}"; do
        while IFS= read -r line; do
            [[ -n "$line" ]] && PDFS+=("$line")
        done < <(walk_root "$r")
    done
fi
if [[ ${#EXPLICIT_PDFS[@]} -gt 0 ]]; then
    for p in "${EXPLICIT_PDFS[@]}"; do
        PDFS+=("$p")
    done
fi

total=${#PDFS[@]}
if [[ $total -eq 0 ]]; then
    echo "No PDFs found. Supply paths or set --root."
    exit 1
fi

echo "Exporting paths for $total airport PDFs → $CORPUS_DIR"
echo ""

failed=0
processed=0
skipped=0
start_ts=$(date +%s)

for ((i=0; i<total; i++)); do
    pdf="${PDFS[$i]}"
    pct=$(( (i + 1) * 100 / total ))
    elapsed=$(( $(date +%s) - start_ts ))
    printf "[%4d/%d  %3d%%  %ds]  " "$((i + 1))" "$total" "$pct" "$elapsed"

    if process_one "$pdf"; then
        # Distinguish skip vs new — process_one writes "[skip ..." in the
        # message itself, but for counting we re-check.
        airport_code="$(basename "$pdf" | sed 's/-faa\.pdf$//' | tr '[:upper:]' '[:lower:]')"
        if [[ -f "$CORPUS_DIR/$airport_code/${airport_code}_paths.csv" ]]; then
            (( processed++ )) || true
        fi
    else
        echo "   FAILED: $pdf"
        (( failed++ )) || true
    fi
done

elapsed=$(( $(date +%s) - start_ts ))
echo ""
echo "Done. processed=$processed failed=$failed elapsed=${elapsed}s"
[[ $failed -eq 0 ]] || exit 1
