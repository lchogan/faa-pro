#!/usr/bin/env bash
# classify.sh — end-to-end airport diagram classifier.
#
# Usage:
#   bash classify.sh bna-faa.pdf
#   bash classify.sh airports/bna-faa.pdf airports/atl-faa.pdf
#
# The PDF must be named <airport>-faa.pdf.
# Produces <airport>-diagram.ai in the same folder as the PDF.
#
# Requires:
#   - Adobe Illustrator installed and accessible via AppleScript
#   - Python venv at faa-pro/.venv with deps installed
#   - Trained model at faa-pro/python/runs/v24/model.lgb

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT/.venv/bin/python3"
IMPORT_JSX="$PROJECT/ImportPredictedLayers.jsx"
CONFIG="/tmp/classify_config.json"

# ---------------------------------------------------------------------------
# run_jsx <path-to-jsx>
# Tells Illustrator (via AppleScript) to run the given JSX file and waits
# for it to finish before returning.
# Reads file content directly — avoids #include reliability issues in AI 2022+.
# ---------------------------------------------------------------------------
run_jsx() {
    local jsx="$1"
    # Pass the path as an argv argument so the heredoc stays single-quoted
    # (no variable expansion) and the path doesn't need escaping.
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
# ---------------------------------------------------------------------------
process_one() {
    local pdf
    # Resolve to absolute path so JSX can find it regardless of working dir.
    pdf="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

    local airport
    airport="$(basename "$pdf" | sed 's/-faa\.pdf$//' | tr '[:upper:]' '[:lower:]')"

    local folder
    folder="$(dirname "$pdf")"

    local paths_csv="$folder/${airport}_paths.csv"
    local edges_csv="$folder/${airport}_paths_edges.csv"
    local json_out="$folder/${airport}_predictions.json"
    local diagram_ai="$folder/${airport}-diagram.ai"

    echo ""
    echo "▶  $airport"

    # Write config for the import-step JSX. ready_ai is intentionally
    # absent — the ImportPredictedLayers JSX falls back to opening the
    # source PDF directly when there's no -ready.ai (PyMuPDF pipeline).
    cat > "$CONFIG" <<JSON
{
  "pdf_path":   "$pdf",
  "airport":    "$airport",
  "folder":     "$folder",
  "paths_csv":  "$paths_csv",
  "edges_csv":  "$edges_csv",
  "json_path":  "$json_out",
  "diagram_ai": "$diagram_ai"
}
JSON

    # ------------------------------------------------------------------
    # Step 1 — PyMuPDF: extract path CSV (no Illustrator)
    # ------------------------------------------------------------------
    echo "   [1/3] PyMuPDF: extract paths..."
    "$PYTHON" "$PROJECT/python/extract_paths_fitz.py" \
        --pdf "$pdf" --airport "$airport" --out-dir "$folder" >/dev/null

    if [[ ! -f "$paths_csv" ]]; then
        echo "   ERROR: PyMuPDF extract produced no CSV."
        return 1
    fi
    local path_count
    path_count=$(( $(wc -l < "$paths_csv") - 1 ))
    echo "   ✓ extracted $path_count paths"

    # ------------------------------------------------------------------
    # Step 2 — Python: build features, run model + override
    # ------------------------------------------------------------------
    echo "   [2/3] Python: classify paths..."
    "$PYTHON" "$PROJECT/python/predict_one.py" \
        --paths "$paths_csv" \
        --airport-folder "$folder" \
        --out "$json_out"

    if [[ ! -f "$json_out" ]]; then
        echo "   ERROR: Python produced no predictions JSON."
        return 1
    fi
    echo "   ✓ predictions written"

    # ------------------------------------------------------------------
    # Step 3 — Illustrator: open PDF, apply predicted layers (matched to
    #          predictions by bbox, not object_id), save diagram.ai
    # ------------------------------------------------------------------
    echo "   [3/3] Illustrator: apply layers + save diagram..."
    run_jsx "$IMPORT_JSX"

    if [[ ! -f "$diagram_ai" ]]; then
        echo "   ERROR: ImportPredictedLayers produced no diagram.ai."
        return 1
    fi

    # Clean up all intermediates.
    rm -f "$paths_csv" "$edges_csv" "$json_out" "$CONFIG"

    echo "   ✓ saved: $diagram_ai"
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    echo "Usage: bash classify.sh <airport>-faa.pdf [<airport>-faa.pdf ...]"
    exit 1
fi

failed=0
for pdf_arg in "$@"; do
    if ! process_one "$pdf_arg"; then
        echo "   FAILED: $pdf_arg"
        (( failed++ )) || true
    fi
done

echo ""
if [[ $failed -eq 0 ]]; then
    echo "Done. All airports classified."
else
    echo "$failed airport(s) failed."
    exit 1
fi
