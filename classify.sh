#!/usr/bin/env bash
# classify.sh — end-to-end airport diagram classifier.
#
# Usage:
#   bash classify.sh bna-faa.pdf
#   bash classify.sh airports/bna-faa.pdf airports/atl-faa.pdf
#
# The PDF must be named <airport>-faa.pdf.
# Produces <airport>-diagram.svg in the same folder as the PDF. Open the
# SVG in Illustrator (File > Open) — top-level groups are tagged as
# Inkscape layers, which Illustrator's SVG importer turns into native
# layers. Save As .ai if you want the .ai extension.
#
# Requires:
#   - Python venv at faa-pro/.venv with deps installed
#   - Trained model at faa-pro/python/runs/v25/model.lgb
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

# Extra args appended to classify_pipeline.py — set by env to override.
# Default disables hull rejection (see header comment for rationale).
PIPELINE_EXTRA="${PIPELINE_EXTRA:---skip-hull}"

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
    # Step 2 — classify_pipeline.py runs the full 6-step pipeline in
    #          order: taxi surfaces (1), runways (2), runway+taxi
    #          labels (3), hull rejection [SKIPPED in v1] (4), ML on
    #          remaining unclaimed (5), stroked-only sweep (6). ML
    #          never sees polygons claimed in steps 1-3.
    # ------------------------------------------------------------------
    echo "   [2/3] Pipeline: 6-step classification..."
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
    echo "   [3/3] Render layered SVG..."
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
