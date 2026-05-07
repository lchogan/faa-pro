/**
 * ImportPredictedLayers.jsx
 *
 * Reads a predictions.json file produced by python/predict.py and moves each
 * vector object in the open document to its predicted layer
 * (Taxiways / Footprints / Runways / Lights / Other).
 *
 * Object identity is established by re-walking the document in the SAME order
 * that ExportClassifiedPaths.jsx walked it (compound paths first by document
 * order, then path items whose parent is not a compound path). The resulting
 * object_id sequence must match the one in the JSON.
 *
 * Usage:
 *   1. Open the unlabeled diagram in Illustrator (the same one you ran
 *      ExportClassifiedPaths.jsx on).
 *   2. File > Scripts > Other Script... > select this file.
 *   3. Pick the predictions.json file.
 */

var CONFIG_FILE = "/tmp/classify_config.json";

// Confident classes (top half of the Layers panel) and Maybe classes (below).
// Each Maybe layer holds path items the model wasn't fully confident about —
// scan one Maybe layer at a time to confirm or reclassify.
// Final output layers in panel order (top → bottom). The override
// pipeline guarantees deterministic placement for:
//   Lights        — every stroked-only polygon, no model
//   Taxiways      — every gray-filled polygon, no model
//   Taxiway Labels — text matches taxiway pattern + group centroid sits
//                   inside actual taxiway pavement (point-in-polygon)
//   Runway Labels — text matches NASR runway designation + group centroid
//                   has low lateral offset to a runway centerline
// Other classes come from the model. "Maybe X" never appears.
// "Metadata" is the renamed default Layer 1 (items that matched no
// prediction — chart border, date stamp, etc.).
var TARGET_LAYERS = [
    "Lights",
    "Footprints",
    "Runways",
    "Runway Labels",
    "Taxiways",
    "Taxiway Labels",
    "Stars",
    "Other",
    "Metadata"
];

function main() {
    // Automated mode: classify.sh wrote a config — skip all dialogs.
    var config = readConfig();
    var isAutomated = (config !== null);

    var doc;
    var jsonFile;
    var diagramAi = null;

    if (isAutomated) {
        // Prefer the source PDF (PyMuPDF-extraction pipeline) — open it
        // directly. Fall back to a -ready.ai if the legacy JSX-extraction
        // pipeline was used.
        doc = null;
        var sourcePath = config.pdf_path || config.ready_ai;
        if (!sourcePath) { return; }
        for (var d = 0; d < app.documents.length; d++) {
            try {
                if (app.documents[d].fullName.fsName === sourcePath) {
                    doc = app.documents[d];
                    break;
                }
            } catch (ignore) {}
        }
        if (!doc) {
            var srcFile = new File(sourcePath);
            if (!srcFile.exists) { return; }
            try {
                app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
                try {
                    var pdfOpts = app.preferences.PDFFileOptions;
                    pdfOpts.pageToOpen = 1;
                } catch (eopts) {}
                doc = app.open(srcFile, DocumentColorSpace.RGB);
            } catch (eopen) {
                return;
            }
        }
        app.activeDocument = doc;
        jsonFile = new File(config.json_path);
        diagramAi = config.diagram_ai;
    } else {
        if (app.documents.length === 0) {
            alert("Open the diagram in Illustrator first.");
            return;
        }
        doc = app.activeDocument;
        jsonFile = File.openDialog("Select predictions.json", "*.json");
        if (!jsonFile) return;
    }

    var payload = readJSON(jsonFile);
    if (!payload) return;
    if (!payload.predictions || payload.predictions.length === 0) {
        alert("predictions.json contains no predictions.");
        return;
    }

    // Build object_id -> label lookup
    // Detect prediction format. The new pipeline (PyMuPDF extraction)
    // emits a `left/top/right/bottom` bbox per prediction; we match each
    // Illustrator path/compound to its prediction by bbox-center proximity.
    // The old pipeline (JSX extraction) used object_id ordering; we keep
    // that path as a fallback for predictions that don't have bboxes.
    var hasBboxes = (payload.predictions.length > 0
        && payload.predictions[0].left !== undefined);

    var preds = [];
    for (var p = 0; p < payload.predictions.length; p++) {
        var pr = payload.predictions[p];
        if (hasBboxes) {
            preds.push({
                cx: (pr.left + pr.right) / 2,
                cy: (pr.top + pr.bottom) / 2,
                w: pr.right - pr.left,
                h: pr.top - pr.bottom,
                label: pr.label,
                used: false
            });
        }
    }
    var idToLabel = {};
    if (!hasBboxes) {
        for (var p2 = 0; p2 < payload.predictions.length; p2++) {
            idToLabel[payload.predictions[p2].object_id] = payload.predictions[p2].label;
        }
    }

    // Bbox-center match: for an Illustrator item with bounds [L,T,R,B] (T>B
    // since AI is y-up), find the unused prediction whose center is closest
    // to the item's center and whose dimensions are within 25% of it. The
    // 25% slack absorbs tiny PyMuPDF-vs-Illustrator rounding.
    function findBboxMatch(bnds) {
        var iCx = (bnds[0] + bnds[2]) / 2;
        var iCy = (bnds[1] + bnds[3]) / 2;
        var iW = bnds[2] - bnds[0];
        var iH = bnds[1] - bnds[3];
        var bestIdx = -1;
        var bestDist = 1e9;
        for (var i = 0; i < preds.length; i++) {
            if (preds[i].used) continue;
            var dx = preds[i].cx - iCx;
            var dy = preds[i].cy - iCy;
            var dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > 2.0) continue;  // PDF/AI coords should agree to <<1pt
            var wEps = 0.25 * Math.max(iW, 1);
            var hEps = 0.25 * Math.max(iH, 1);
            if (Math.abs(preds[i].w - iW) > wEps) continue;
            if (Math.abs(preds[i].h - iH) > hEps) continue;
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    var prevAlerts = app.userInteractionLevel;
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

    // Illustrator imports a PDF into a single "Layer 1" by default. After
    // we move identified items off it, what remains is chart metadata
    // (page border, date stamp). Rename the layer so its final name
    // reflects what it contains.
    for (var li = 0; li < doc.layers.length; li++) {
        if (doc.layers[li].name === "Layer 1") {
            doc.layers[li].name = "Metadata";
            break;
        }
    }

    var layers = ensureLayers(doc, TARGET_LAYERS);
    var stats = {};
    for (var i = 0; i < TARGET_LAYERS.length; i++) stats[TARGET_LAYERS[i]] = 0;
    var skipped = 0;
    var unmatched = 0;
    var objectId = 0;

    function applyLabel(item, label) {
        // Predictions referencing a layer not in TARGET_LAYERS (e.g. a
        // residual "Maybe X" from older predictions JSON) get routed to
        // Other so nothing ends up un-categorized.
        if (!layers[label]) { label = "Other"; }
        if (!layers[label]) { skipped++; return; }
        try {
            item.move(layers[label], ElementPlacement.PLACEATEND);
            stats[label]++;
        } catch (e) { skipped++; }
    }

    // Pass 1: compound paths
    var compounds = doc.compoundPathItems;
    var cpList = [];
    for (var c = 0; c < compounds.length; c++) cpList.push(compounds[c]);
    for (var i2 = 0; i2 < cpList.length; i2++) {
        var cp = cpList[i2];
        if (hasBboxes) {
            var idx = findBboxMatch(cp.geometricBounds);
            if (idx >= 0) {
                preds[idx].used = true;
                applyLabel(cp, preds[idx].label);
            } else {
                unmatched++;
            }
        } else {
            var label = idToLabel[objectId];
            objectId++;
            if (label === undefined) { unmatched++; continue; }
            applyLabel(cp, label);
        }
    }

    // Pass 2: path items whose parent is not a compound path
    var paths = doc.pathItems;
    var pathList = [];
    for (var k = 0; k < paths.length; k++) {
        if (paths[k].parent && paths[k].parent.typename === "CompoundPathItem") continue;
        pathList.push(paths[k]);
    }
    for (var k2 = 0; k2 < pathList.length; k2++) {
        var pp = pathList[k2];
        if (hasBboxes) {
            var idx2 = findBboxMatch(pp.geometricBounds);
            if (idx2 >= 0) {
                preds[idx2].used = true;
                applyLabel(pp, preds[idx2].label);
            } else {
                unmatched++;
            }
        } else {
            var label2 = idToLabel[objectId];
            objectId++;
            if (label2 === undefined) { unmatched++; continue; }
            applyLabel(pp, label2);
        }
    }

    // Reorder layers to match TARGET_LAYERS top-to-bottom. We bring each
    // layer to the front in REVERSE iteration order, so the first entry
    // in TARGET_LAYERS ends up topmost and the last entry (Metadata)
    // ends up at the bottom.
    for (var ord = TARGET_LAYERS.length - 1; ord >= 0; ord--) {
        var lyr = layers[TARGET_LAYERS[ord]];
        if (lyr) {
            try { lyr.zOrder(ZOrderMethod.BRINGTOFRONT); } catch (eOrd) {}
        }
    }

    app.userInteractionLevel = prevAlerts;

    if (isAutomated && diagramAi) {
        var saveOpts = new IllustratorSaveOptions();
        saveOpts.compatibility = Compatibility.ILLUSTRATOR17;
        saveOpts.pdfCompatible = true;
        doc.saveAs(new File(diagramAi), saveOpts);
        // Leave the document open so the user can review it immediately.
        return;
    }

    var totalSeen = objectId;
    var totalPredicted = payload.predictions.length;
    var msg = "Layer assignment complete.\n\nObjects processed: " + totalSeen +
              "\nPredictions in JSON: " + totalPredicted + "\n\n";
    for (var t = 0; t < TARGET_LAYERS.length; t++) {
        msg += "  " + TARGET_LAYERS[t] + ": " + stats[TARGET_LAYERS[t]] + "\n";
    }
    if (unmatched > 0) msg += "\nUnmatched (no prediction for object_id): " + unmatched;
    if (skipped > 0) msg += "\nSkipped (move failed or unknown label): " + skipped;
    if (totalSeen !== totalPredicted) {
        msg += "\n\n⚠ Object count mismatch — the document may have been modified " +
               "between export and import. Re-run the export.";
    }
    alert(msg);
}

function ensureLayers(doc, names) {
    var out = {};
    for (var i = 0; i < names.length; i++) {
        var name = names[i];
        var layer = null;
        for (var j = 0; j < doc.layers.length; j++) {
            if (doc.layers[j].name === name) { layer = doc.layers[j]; break; }
        }
        if (!layer) {
            layer = doc.layers.add();
            layer.name = name;
        }
        out[name] = layer;
    }
    return out;
}

function readConfig() {
    var f = new File(CONFIG_FILE);
    if (!f.exists) return null;
    return readJSON(f);
}

function readJSON(file) {
    file.encoding = "UTF-8";
    if (!file.open("r")) {
        alert("Could not open JSON file:\n" + file.fsName);
        return null;
    }
    var text = file.read();
    file.close();
    try {
        // Avoid eval: ExtendScript JSON is built-in in modern Illustrator.
        if (typeof JSON !== "undefined" && JSON.parse) {
            return JSON.parse(text);
        }
        // Fallback for very old AI versions
        return (new Function("return (" + text + ")"))();
    } catch (e) {
        alert("Failed to parse JSON: " + e.message);
        return null;
    }
}

main();
