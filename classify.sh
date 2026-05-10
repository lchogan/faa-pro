#!/usr/bin/env bash
# classify.sh — end-to-end airport diagram classifier.
#
# Usage:
#   bash classify.sh bna-faa.pdf
#   bash classify.sh airports/bna-faa.pdf airports/atl-faa.pdf
#
# The PDF must be named <airport>-faa.pdf.
# Produces <airport>-diagram.svg in the same folder as the PDF. For
# single-airport invocations the SVG is then opened in Illustrator,
# saved as <airport>-diagram.ai, and reorganized via
# PrepareForInspection.jsx (sublayers promoted out of "Layer 1",
# layers sorted into the standard inspection stack, reference-only
# layers locked + hidden). The .ai is left open for review. Set
# OPEN_AFTER_CLASSIFY=0 to skip the Illustrator step; batch invocations
# (more than one PDF) skip it automatically.
#
# Requires:
#   - Python venv at faa-pro/.venv with deps installed
#   - Trained model at faa-pro/python/ml/runs/v25/model.lgb
#
# Pipeline defaults (v1, locked 2026-05-07):
#   --skip-hull           — concave-hull rejection is OFF in production.
#                           Validated on ARB/APF/COS/ELM/F45/MCO/OGG/ORD;
#                           the v25 model's morphology features handle
#                           text/symbol rejection without it, and many
#                           legitimate building footprints (e.g. ARB
#                           FIRE STATION) sit outside the hull.
#   no --footprint-threshold
#                         — argmax decision rule. Lower thresholds
#                           (0.10, 0.07) catch more borderline buildings
#                           but also pull in arrow symbols on dense
#                           charts like ORD. Pass via PIPELINE_EXTRA env
#                           var if you want to experiment.

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT/.venv/bin/python3"

# Make `from pipeline.* import …` and `from ml.* import …` resolve from
# any script under python/. Required because the pipeline/, ml/, and
# _deprecated/ subpackages aren't installed — they're sibling
# directories under python/, and Python only auto-adds the *script's*
# directory to sys.path. Setting PYTHONPATH here makes python/ the
# resolution root for every invocation in this file.
export PYTHONPATH="$PROJECT/python${PYTHONPATH:+:$PYTHONPATH}"

# Extra args appended to classify_pipeline.py — set by env to override.
# Default disables hull rejection (see header comment for rationale).
PIPELINE_EXTRA="${PIPELINE_EXTRA:---skip-hull}"

# Single-airport invocations get auto-opened in Illustrator. Batch
# invocations (>1 arg) skip the open step to avoid a window per airport.
# Set OPEN_AFTER_CLASSIFY=0 to disable, OPEN_AFTER_CLASSIFY=1 to force on.
ARG_COUNT=$#
if [[ -z "${OPEN_AFTER_CLASSIFY:-}" ]]; then
    if [[ $ARG_COUNT -eq 1 ]]; then
        OPEN_AFTER_CLASSIFY=1
    else
        OPEN_AFTER_CLASSIFY=0
    fi
fi

# ---------------------------------------------------------------------------
# prepare_for_inspection <svg-path>
# Opens the SVG in Illustrator, runs PrepareForInspection.jsx, leaves the
# resulting .ai open. Driven by a small JSON config the JSX reads.
# ---------------------------------------------------------------------------
prepare_for_inspection() {
    local svg="$1"
    local jsx="$PROJECT/PrepareForInspection.jsx"
    local config="/tmp/faa_pro_inspection.json"

    if [[ ! -f "$jsx" ]]; then
        echo "   WARN: $jsx not found — skipping Illustrator open."
        return 0
    fi

    cat > "$config" <<JSON
{
  "svg_path": "$svg"
}
JSON

    osascript - "$jsx" <<'APPLESCRIPT' >/dev/null
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
    local pdf_text_csv="$folder/${airport}_predictions.pdf_text.csv"
    local diagram_svg="$folder/${airport}-diagram.svg"

    echo ""
    echo "▶  $airport"

    # ------------------------------------------------------------------
    # Step 1 — PyMuPDF: extract path CSV (no Illustrator)
    # ------------------------------------------------------------------
    echo "   [1/4] PyMuPDF: extract paths..."
    "$PYTHON" "$PROJECT/python/pipeline/extract_paths_fitz.py" \
        --pdf "$pdf" --airport "$airport" --out-dir "$folder" >/dev/null

    if [[ ! -f "$paths_csv" ]]; then
        echo "   ERROR: PyMuPDF extract produced no CSV."
        return 1
    fi
    local path_count
    path_count=$(( $(wc -l < "$paths_csv") - 1 ))
    echo "   ✓ extracted $path_count paths"

    # ------------------------------------------------------------------
    # Step 2 — classify_pipeline.py runs the full pipeline in order:
    #          1   taxi surfaces (gray fill)
    #          2   runways (NASR-driven)
    #          3   runway labels (centerline-token search)
    #          3b  runway-label move along centerline (layout)
    #          4   taxi labels (K-nearest)
    #          5   hull rejection (SKIPPED in production via --skip-hull)
    #          6   ML on remaining unclaimed
    #          7   stroked-only sweep → Other
    #          ML never sees polygons claimed in steps 1-4.
    # ------------------------------------------------------------------
    echo "   [2/4] Pipeline: classification..."
    "$PYTHON" "$PROJECT/python/classify_pipeline.py" \
        --paths "$paths_csv" \
        --pdf "$pdf" \
        --out "$json_out" \
        $PIPELINE_EXTRA

    if [[ ! -f "$json_out" ]]; then
        echo "   ERROR: Pipeline produced no predictions JSON."
        return 1
    fi
    echo "   ✓ predictions written"

    # ------------------------------------------------------------------
    # Step 3 — Render layered SVG. Each layer is a top-level <g> tagged
    #          with Inkscape layer attributes; Illustrator's SVG
    #          importer turns each into a native AI layer. ~1s vs the
    #          old JSX route's 60+s.
    # ------------------------------------------------------------------
    echo "   [3/4] Render layered SVG..."
    "$PYTHON" "$PROJECT/python/render_svg_layers.py" \
        --pdf "$pdf" \
        --predictions "$json_out" \
        --out "$diagram_svg" >/dev/null

    if [[ ! -f "$diagram_svg" ]]; then
        echo "   ERROR: SVG renderer produced no output."
        return 1
    fi

    # Clean up intermediates. Predictions JSON kept for debugging.
    rm -f "$paths_csv" "$edges_csv" "$pdf_text_csv"

    echo "   ✓ saved: $diagram_svg"

    # ------------------------------------------------------------------
    # Step 4 — Open in Illustrator and reorganize layers for inspection.
    #          Saves as <airport>-diagram.ai, promotes SVG-import
    #          sublayers out of "Layer 1", locks the reference-only
    #          layers, and leaves the .ai open. Skipped on batch runs.
    # ------------------------------------------------------------------
    if [[ "$OPEN_AFTER_CLASSIFY" == "1" ]]; then
        echo "   [4/4] Reorganize layers in Illustrator..."
        prepare_for_inspection "$diagram_svg"
        echo "   ✓ open in Illustrator: ${airport}-diagram.ai"
    else
        echo "   [4/4] Skipping Illustrator open (batch run or OPEN_AFTER_CLASSIFY=0)."
    fi
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
