/**
 * PrepareForLabeling.jsx
 *
 * Prepares a fresh FAA airport PDF for clean manual labeling. Opens the PDF,
 * releases clipping masks, deletes off-artboard junk, and scaffolds the
 * seven target layers plus a notes layer. Everything currently in the
 * document is moved to a layer named "Unclassified" — the user's job is to
 * drag each object into the correct labeled layer.
 *
 * IMPORTANT — RULES FOR THE LABELER:
 *   - DO NOT use Pathfinder (Unite, Merge, etc.) to combine shapes.
 *   - DO NOT change fill or stroke colors.
 *   - DO NOT rotate, scale, or move artwork.
 *   - DO NOT modify the artboard.
 *   - DO NOT use Live Paint, Image Trace, or any operation that reshapes paths.
 *   - The geometry must remain identical to what the classifier sees on a
 *     fresh PDF — only the layer assignment is changing.
 *
 * Workflow:
 *   1. File > Scripts > Other Script > select this file.
 *   2. Pick the FAA source PDF (e.g. 00079ad.pdf).
 *   3. Enter the airport code (e.g. "cha"). The script saves as
 *      <code>-diagram.ai next to the PDF.
 *   4. Use Window > Layers to drag each object to the correct layer.
 *   5. Save when done. Run ExportClassifiedPaths.jsx later for training.
 */

var TARGET_LAYERS = [
    "Taxiways",
    "Footprints",
    "Runways",
    "Lights",
    "Taxiway Labels",
    "Runway Labels",
    "Other"
];

var NOTES_LAYER = "LABELING_NOTES (do not modify)";
var HOLDING_LAYER = "Unclassified";

var NOTES_TEXT =
    "LABELING RULES — geometry must remain unchanged.\n" +
    "\n" +
    "  • Do NOT use Pathfinder (Unite/Merge/etc).\n" +
    "  • Do NOT change fill or stroke colors.\n" +
    "  • Do NOT rotate, scale, or move artwork.\n" +
    "  • Do NOT modify the artboard.\n" +
    "  • Drag each object from \"" + HOLDING_LAYER + "\" into the correct layer.\n" +
    "  • Empty out \"" + HOLDING_LAYER + "\" before saving (everything else → Other).";

function main() {
    var pdfFile = File.openDialog("Select FAA airport PDF", "*.pdf");
    if (!pdfFile) return;

    var code = prompt("Airport code (e.g. cha, atl, ord):", "");
    if (!code) return;
    code = String(code).replace(/[^A-Za-z0-9_-]/g, "").toLowerCase();
    if (!code) { alert("Airport code is empty — aborting."); return; }

    var prevAlerts = app.userInteractionLevel;
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

    var doc = app.open(pdfFile);

    try {
        releaseAllClippingMasks(doc);
        deleteOffArtboardItems(doc);

        renameInitialLayer(doc, HOLDING_LAYER);
        ensureLayer(doc, NOTES_LAYER);
        for (var i = 0; i < TARGET_LAYERS.length; i++) {
            ensureLayer(doc, TARGET_LAYERS[i]);
        }

        // Re-order: notes on top, then targets, then holding.
        // doc.layers behaves like a top-of-stack array; index 0 is topmost.
        // Move HOLDING to bottom, NOTES to top.
        moveLayerToBottom(doc, HOLDING_LAYER);
        for (var j = TARGET_LAYERS.length - 1; j >= 0; j--) {
            moveLayerToBottom(doc, TARGET_LAYERS[j]);
        }
        moveLayerToTop(doc, NOTES_LAYER);

        // Move HOLDING back below all targets but above nothing.
        moveLayerToBottom(doc, HOLDING_LAYER);

        addNotesText(doc, NOTES_LAYER);

        var outFile = new File(pdfFile.path + "/" + code + "-diagram.ai");
        var saveOpts = new IllustratorSaveOptions();
        saveOpts.compatibility = Compatibility.ILLUSTRATOR17;
        saveOpts.pdfCompatible = true;
        doc.saveAs(outFile, saveOpts);

        app.userInteractionLevel = prevAlerts;

        alert(
            "Ready to label.\n\n" +
            "Saved: " + outFile.fsName + "\n\n" +
            "Open the Layers panel (Window > Layers) and drag each object\n" +
            "from \"" + HOLDING_LAYER + "\" into the correct labeled layer.\n\n" +
            "DO NOT modify geometry, colors, artboard, or transform anything.\n" +
            "Save when finished."
        );
    } catch (e) {
        app.userInteractionLevel = prevAlerts;
        alert("Failed during scaffolding: " + e.message);
    }
}

function releaseAllClippingMasks(doc) {
    var changed = true;
    var safety = 20;
    while (changed && safety-- > 0) {
        changed = false;
        var groups = doc.groupItems;
        for (var i = groups.length - 1; i >= 0; i--) {
            var g = groups[i];
            if (g.clipped) {
                try {
                    app.executeMenuCommand("selectall");
                    app.executeMenuCommand("releaseMask");
                } catch (e) {}
                changed = true;
                break;
            }
        }
    }
    try { doc.selection = null; } catch (e) {}
}

function deleteOffArtboardItems(doc) {
    if (doc.artboards.length === 0) return;
    var ab = doc.artboards[0].artboardRect; // [left, top, right, bottom]
    var pads = doc.pageItems;
    for (var i = pads.length - 1; i >= 0; i--) {
        var p = pads[i];
        var b;
        try { b = p.geometricBounds; } catch (e) { continue; }
        // off-artboard if entirely outside the artboard rect
        if (b[2] < ab[0] || b[0] > ab[2] || b[3] > ab[1] || b[1] < ab[3]) {
            try { p.remove(); } catch (e) {}
        }
    }
}

function renameInitialLayer(doc, newName) {
    if (doc.layers.length === 0) {
        var layer = doc.layers.add();
        layer.name = newName;
        return layer;
    }
    // Default opened-PDF layer is usually called "Layer 1".
    var existing = doc.layers[0];
    if (existing.name !== newName) existing.name = newName;
    return existing;
}

function ensureLayer(doc, name) {
    for (var i = 0; i < doc.layers.length; i++) {
        if (doc.layers[i].name === name) return doc.layers[i];
    }
    var layer = doc.layers.add();
    layer.name = name;
    return layer;
}

function findLayer(doc, name) {
    for (var i = 0; i < doc.layers.length; i++) {
        if (doc.layers[i].name === name) return doc.layers[i];
    }
    return null;
}

function moveLayerToBottom(doc, name) {
    var l = findLayer(doc, name);
    if (!l) return;
    l.zOrder(ZOrderMethod.SENDTOBACK);
}

function moveLayerToTop(doc, name) {
    var l = findLayer(doc, name);
    if (!l) return;
    l.zOrder(ZOrderMethod.BRINGTOFRONT);
}

function addNotesText(doc, layerName) {
    var layer = findLayer(doc, layerName);
    if (!layer) return;
    var ab = doc.artboards[0].artboardRect;
    var x = ab[0] + 18;
    var y = ab[1] - 18;
    try {
        var tf = layer.textFrames.pointText([x, y]);
        tf.contents = NOTES_TEXT;
        tf.textRange.characterAttributes.size = 9;
        layer.locked = true;
    } catch (e) {}
}

main();
