/**
 * ClassifyAirport.jsx — one-shot pipeline from FAA PDF to classified AI file.
 *
 * What it does (per raw *-faa.pdf):
 *   1. Open the PDF in Illustrator.
 *   2. Release clipping masks and delete off-artboard items.
 *   3. Export the per-path features and per-edge data as CSVs (temp).
 *   4. Shell out to predict_one.py — extracts PDF text, computes features,
 *      runs the trained LightGBM model, applies the FAA NASR runway-name
 *      override, returns predictions.json with confidence-banded
 *      "Maybe X" routing.
 *   5. Create the 15 target layers (8 confident + 7 Maybe).
 *   6. Move each path/compound into the predicted layer.
 *   7. Save as <code>-diagram.ai next to the source PDF.
 *
 * Usage:
 *   File > Scripts > Other Script > ClassifyAirport.jsx
 *   Pick one <code>-faa.pdf to test. If you cancel the file picker, you can
 *   choose a folder for batch mode; batch mode still processes PDFs only.
 *
 * Requires:
 *   - Python venv at faa-pro/.venv/bin/python3 with all deps installed
 *   - Trained model at faa-pro/python/runs/v24/model.lgb
 *   - data/nasr_apt_rwy.csv
 *
 * Hard-coded project paths near the top — edit if your layout differs.
 */

// ================================
// CONFIGURATION
// ================================

var FAA_PRO       = "/Users/lukehogan/AOA-Code/faa-pro";
var PYTHON_BIN    = FAA_PRO + "/.venv/bin/python3";
var PREDICT_PY    = FAA_PRO + "/python/predict_one.py";
var PDF_TO_SVG_PY = FAA_PRO + "/python/pdf_page_to_svg.py";
var TMP_DIR       = "/tmp";  // overridden to airport folder in automated mode
var RELEASE_CLIPPING_MASKS = true;
var CONFIG_FILE   = "/tmp/classify_config.json";

var TARGET_LAYERS = [
    "Taxiways", "Footprints", "Runways", "Lights",
    "Taxiway Labels", "Runway Labels", "Stars", "Other",
    "Maybe Taxiway", "Maybe Footprint", "Maybe Runway", "Maybe Light",
    "Maybe Taxiway Label", "Maybe Runway Label", "Maybe Star"
];

// CSV columns — must match ExportClassifiedPaths.jsx exactly.
var COLUMNS = [
    "airport","object_id","kind","source_layer","label",
    "left","top","right","bottom","width","height",
    "bbox_area","poly_area","perimeter",
    "centroid_x","centroid_y","aspect",
    "num_anchors","subpath_count","closed",
    "filled","fill_kind","fill_r","fill_g","fill_b",
    "stroked","stroke_kind","stroke_r","stroke_g","stroke_b","stroke_width",
    "principal_angle","principal_ratio",
    "longest_segment_angle","longest_segment_length",
    "artboard_left","artboard_top","artboard_right","artboard_bottom"
];
var EDGE_COLUMNS = [
    "airport","object_id","subpath_index","edge_index",
    "mid_x","mid_y","angle","length"
];

// ================================
// MAIN
// ================================

function main() {
    try { app.preferences.setBooleanPreference("text/preferenceShowMissingFontsDialog", false); } catch (e) {}
    try { app.preferences.setBooleanPreference("ShowExternalJSXWarning", false); } catch (e) {}

    var prevAlerts = app.userInteractionLevel;

    // Automated mode: classify.sh wrote a config file — skip all dialogs.
    var config = readClassifyConfig();
    var isAutomated = (config !== null);

    // Export-only mode: skip Python prediction + layer-import + save. Used by
    // export_paths.sh during distant-supervision corpus building.
    var exportOnly = isAutomated && !!config.export_only;
    var saveReadyOnly = false;

    var sources = [];
    if (isAutomated) {
        var autoPdf = new File(config.pdf_path);
        if (!autoPdf.exists) { return; } // shell script will detect missing CSV
        sources.push(autoPdf);
        TMP_DIR = config.folder; // intermediates go next to the PDF, not /tmp
    } else {
        var pdfFile = File.openDialog("Select one FAA airport PDF (*-faa.pdf)", "*.pdf");
        if (pdfFile) {
            if (!/-faa\.pdf$/i.test(pdfFile.name)) {
                alert("Please select a file named like <airport>-faa.pdf.");
                return;
            }
            sources.push(pdfFile);
        } else {
            var rootFolder = Folder.selectDialog("Or select airport folder / parent folder for PDF batch");
            if (!rootFolder) return;
            sources = findAirportSources(rootFolder);
            if (sources.length === 0) {
                alert("No *-faa.pdf files found under:\n" + rootFolder.fsName);
                return;
            }
        }
    }

    var processed = 0;
    var errors = [];
    var startTime = new Date().getTime();

    // In automated mode, classify.sh runs Python — force export-only regardless
    // of whether system.callSystem is available.
    var hasCallSystem = !isAutomated && (typeof system !== "undefined" && !!system.callSystem);
    var shCommands = [];   // populated in export-only manual mode
    var isBatch = sources.length > 1;

    for (var i = 0; i < sources.length; i++) {
        var src = sources[i];
        try {
            processOne(src, hasCallSystem, shCommands, isBatch, exportOnly, saveReadyOnly, config);
            processed++;
        } catch (e) {
            errors.push(src.name + ": " + e.message);
            try { app.activeDocument.close(SaveOptions.DONOTSAVECHANGES); } catch (ignore) {}
        }
        app.userInteractionLevel = prevAlerts;
    }

    app.userInteractionLevel = prevAlerts;
    var elapsed = Math.round((new Date().getTime() - startTime) / 1000);
    var msg = "Classification complete.\n\n" +
              "Processed: " + processed + " / " + sources.length + "\n" +
              "Elapsed: " + elapsed + "s";
    if (errors.length > 0) {
        msg += "\n\nErrors:\n  " + errors.slice(0, 10).join("\n  ");
        if (errors.length > 10) msg += "\n  ... (+" + (errors.length - 10) + " more)";
    }

    if (!isAutomated && shCommands.length > 0) {
        // Manual mode, no system.callSystem — write a shell script and instruct user.
        var shFile = new File("/tmp/run_predictions.sh");
        shFile.encoding = "UTF-8";
        shFile.lineFeed = "Unix";  // ExtendScript defaults to \r; bash needs \n
        shFile.open("w");
        shFile.writeln("#!/bin/bash");
        shFile.writeln("set -e");
        for (var si = 0; si < shCommands.length; si++) shFile.writeln(shCommands[si]);
        shFile.writeln("echo 'All predictions complete.'");
        shFile.close();

        msg += "\n\n---\nsystem.callSystem is not available in this Illustrator version.\n" +
               "Preprocessing is done. Next steps:\n\n" +
               "1. Open Terminal and run:\n" +
               "     bash /tmp/run_predictions.sh\n\n";
        if (isBatch) {
            msg += "2. For each *-ready.ai file: open it in Illustrator,\n" +
                   "   then run ImportPredictedLayers.jsx and pick the\n" +
                   "   matching /tmp/<airport>_predictions.json file.";
        } else {
            msg += "2. Switch back to Illustrator (the preprocessed document\n" +
                   "   is already open), then run ImportPredictedLayers.jsx\n" +
                   "   and pick the predictions.json from /tmp/.";
        }
    }

    // In automated mode the shell script handles reporting — no dialog needed.
    if (!isAutomated) {
        alert(msg);
    }
}

/**
 * @param {File}    srcFile       - source *-faa.pdf
 * @param {boolean} hasCallSystem - whether system.callSystem is available
 * @param {Array}   shCommands    - shell commands are pushed here in export-only mode
 * @param {boolean} isBatch       - true when processing more than one PDF
 * @param {boolean} exportOnly    - if true, stop after CSV export (skip Python + save)
 * @param {boolean} saveReadyOnly - if true, export CSV and save -ready.ai then exit
 * @param {Object}  config        - automated-mode config (may contain paths_csv/edges_csv overrides)
 */
function processOne(srcFile, hasCallSystem, shCommands, isBatch, exportOnly, saveReadyOnly, config) {
    var airport = airportCodeFromFilename(srcFile.name);
    var folder = srcFile.parent;
    var aiOut = new File(folder.fsName + "/" + airport + "-diagram.ai");
    // Allow the shell wrapper to redirect CSVs to a corpus folder rather than
    // the airport's own folder. Falls back to TMP_DIR-based defaults.
    var pathsCsv = (config && config.paths_csv) ? config.paths_csv : (TMP_DIR + "/" + airport + "_paths.csv");
    var edgesCsv = (config && config.edges_csv) ? config.edges_csv : (TMP_DIR + "/" + airport + "_paths_edges.csv");
    var jsonOut = TMP_DIR + "/" + airport + "_predictions.json";
    var logPath = TMP_DIR + "/classify_" + airport + ".log";
    var logFile = new File(logPath);
    logFile.encoding = "UTF-8";
    logFile.open("w");
    logFile.writeln("[" + new Date() + "] processing " + srcFile.fsName);

    var step = "init";
    try {
        step = "open";
        logFile.writeln("[" + step + "] " + srcFile.fsName);
        var doc = openPdfDocument(srcFile, logFile);
        app.activeDocument = doc;
        app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

        if (RELEASE_CLIPPING_MASKS) {
            step = "release-masks";
            logFile.writeln("[" + step + "]");
            releaseAllClippingMasks(doc, logFile);
        } else {
            logFile.writeln("[release-masks] skipped for " + srcFile.name);
        }

        step = "delete-off-artboard";
        logFile.writeln("[" + step + "]");
        deleteOffArtboardItems(doc);

        step = "export-paths-edges";
        logFile.writeln("[" + step + "] -> " + pathsCsv);
        exportPathsAndEdges(doc, airport, pathsCsv, edgesCsv);

        if (exportOnly) {
            logFile.writeln("[done-export-only]");
            logFile.close();
            try { doc.close(SaveOptions.DONOTSAVECHANGES); } catch (ignore) {}
            return;
        }

        if (saveReadyOnly) {
            step = "save-ready-only";
            var readyPath = (config && config.ready_ai)
                ? config.ready_ai
                : (folder.fsName + "/" + airport + "-ready.ai");
            var readyFile2 = new File(readyPath);
            var saveOptsReady2 = new IllustratorSaveOptions();
            saveOptsReady2.compatibility = Compatibility.ILLUSTRATOR17;
            saveOptsReady2.pdfCompatible = true;
            doc.saveAs(readyFile2, saveOptsReady2);
            logFile.writeln("[save-ready-only] " + readyFile2.fsName);
            logFile.close();
            try { doc.close(SaveOptions.DONOTSAVECHANGES); } catch (ignore) {}
            return;
        }

        step = "python";
        var cmd = quote(PYTHON_BIN) + " " + quote(PREDICT_PY) +
                  " --paths " + quote(pathsCsv) +
                  " --airport-folder " + quote(folder.fsName) +
                  " --out " + quote(jsonOut) + " 2>&1";
        logFile.writeln("[" + step + "] " + cmd);

        if (!hasCallSystem) {
            // system.callSystem removed in Illustrator 2022+. Save the preprocessed
            // document so ImportPredictedLayers.jsx can apply predictions after the
            // user runs Python externally via /tmp/run_predictions.sh.
            step = "save-ready";
            var readyFile = new File(folder.fsName + "/" + airport + "-ready.ai");
            var saveOptsReady = new IllustratorSaveOptions();
            saveOptsReady.compatibility = Compatibility.ILLUSTRATOR17;
            saveOptsReady.pdfCompatible = true;
            doc.saveAs(readyFile, saveOptsReady);
            logFile.writeln("[save-ready] " + readyFile.fsName);
            logFile.writeln("[done-phase1] run Python then ImportPredictedLayers.jsx");
            logFile.close();
            shCommands.push(cmd);
            // In batch mode close the document; in single-file mode leave it open
            // so the user can run ImportPredictedLayers.jsx without reopening it.
            if (isBatch) doc.close(SaveOptions.DONOTSAVECHANGES);
            return;
        }

        var output = system.callSystem(cmd);
        logFile.writeln("[python output]\n" + output);
        var jsonFile = new File(jsonOut);
        if (!jsonFile.exists) {
            throw new Error("predict_one.py produced no output. Tail:\n" +
                            (output.length > 500 ? output.substring(output.length - 500) : output));
        }

        step = "read-json";
        logFile.writeln("[" + step + "]");
        var preds = readJson(jsonFile);
        if (!preds || !preds.predictions) throw new Error("Bad predictions JSON: " + jsonOut);
        var idToLabel = {};
        for (var p = 0; p < preds.predictions.length; p++) {
            idToLabel[preds.predictions[p].object_id] = preds.predictions[p].label;
        }

        step = "create-layers";
        logFile.writeln("[" + step + "]");
        var layers = ensureLayers(doc, TARGET_LAYERS);

        step = "apply-predictions";
        logFile.writeln("[" + step + "] " + preds.predictions.length + " predictions");
        applyPredictions(doc, idToLabel, layers);

        step = "save-as-ai";
        logFile.writeln("[" + step + "] -> " + aiOut.fsName);
        var saveOpts = new IllustratorSaveOptions();
        saveOpts.compatibility = Compatibility.ILLUSTRATOR17;
        saveOpts.pdfCompatible = true;
        doc.saveAs(aiOut, saveOpts);
        doc.close(SaveOptions.DONOTSAVECHANGES);

        logFile.writeln("[done]");
        logFile.close();
    } catch (e) {
        logFile.writeln("[FAILED at " + step + "] " + e.message);
        logFile.close();
        throw new Error("[" + step + "] " + e.message + "  (log: " + logPath + ")");
    }
}

// ================================
// FILE DISCOVERY
// ================================

// Walks `folder` recursively and returns only raw FAA PDFs. Existing
// `*-diagram.ai` files are training artifacts and are intentionally ignored.
function findAirportSources(folder) {
    var results = [];
    walkAirportSources(folder, results);
    return results;
}

function walkAirportSources(folder, out) {
    var children;
    try { children = folder.getFiles(); } catch (e) { return; }
    var faaPdf = null;
    var subFolders = [];
    for (var i = 0; i < children.length; i++) {
        var c = children[i];
        if (c instanceof Folder) {
            subFolders.push(c);
        } else if (c instanceof File) {
            if (/-faa\.pdf$/i.test(c.name) && !faaPdf) faaPdf = c;
        }
    }
    if (faaPdf) out.push(faaPdf);
    for (var j = 0; j < subFolders.length; j++) walkAirportSources(subFolders[j], out);
}

function airportCodeFromFilename(name) {
    var m = name.match(/^(.+?)-faa\.pdf$/i);
    return m ? m[1].toLowerCase().replace(/[^a-z0-9_-]/g, "") : name.replace(/\.[^.]+$/, "");
}

function openPdfDocument(srcFile, logFile) {
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
    configurePdfOpenOptions(logFile);

    // Opening PDFs with visible import UI can trigger Illustrator's memory
    // error before the import completes. With DONTDISPLAYALERTS, Illustrator
    // should respect app.preferences.PDFFileOptions and open page 1 directly.
    if (logFile) logFile.writeln("[open] DONTDISPLAYALERTS + PDFFileOptions pageToOpen=1");
    try {
        return app.open(srcFile, DocumentColorSpace.RGB);
    } catch (e1) {
        if (logFile) logFile.writeln("[open] app.open(file, RGB) failed: " + e1.message);
    }
    try {
        return open(srcFile, DocumentColorSpace.RGB);
    } catch (e2) {
        if (logFile) logFile.writeln("[open] open(file, RGB) failed: " + e2.message);
    }

    if (logFile) logFile.writeln("[open] falling back to place+embed PDF conversion");
    try {
        return placeAndEmbedPdf(srcFile, logFile);
    } catch (e3) {
        if (logFile) logFile.writeln("[open] place+embed PDF failed: " + e3.message);
    }

    if (logFile) logFile.writeln("[open] falling back to PDF->SVG conversion");
    return openPdfViaSvg(srcFile, logFile);
}

function configurePdfOpenOptions(logFile) {
    try {
        var pdfOptions = app.preferences.PDFFileOptions;
        pdfOptions.pageToOpen = 1;

        var cropBox = null;
        try { cropBox = PDFBoxType.PDFMEDIABOX; } catch (ignore1) {}
        if (cropBox === null) {
            try { cropBox = PDFBoxType.PDFMediaBox; } catch (ignore2) {}
        }
        if (cropBox !== null) pdfOptions.pDFCropToBox = cropBox;
        if (logFile) logFile.writeln("[open] configured app.preferences.PDFFileOptions");
    } catch (e) {
        if (logFile) logFile.writeln("[open] could not configure PDFFileOptions: " + e.message);
    }
}

function placeAndEmbedPdf(srcFile, logFile) {
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
    configurePdfOpenOptions(logFile);

    var doc = app.documents.add(DocumentColorSpace.RGB, 612, 792);
    var placed = doc.placedItems.add();
    var pdfForPlace = new File(srcFile.fsName);
    if (logFile) {
        logFile.writeln("[open] place file exists=" + pdfForPlace.exists +
            " path=" + pdfForPlace.fsName);
    }
    try {
        placed.file = pdfForPlace;
    } catch (e1) {
        if (logFile) logFile.writeln("[open] placed.file File failed: " + e1.message);
        try {
            placed.file = pdfForPlace.fsName;
        } catch (e2) {
            if (logFile) logFile.writeln("[open] placed.file string failed: " + e2.message);
            throw e1;
        }
    }

    var b = placed.geometricBounds; // [left, top, right, bottom]
    try {
        doc.artboards[0].artboardRect = [b[0], b[1], b[2], b[3]];
    } catch (ignore) {}

    if (logFile) {
        logFile.writeln("[open] placed PDF bounds: " +
            round4(b[0]) + "," + round4(b[1]) + "," + round4(b[2]) + "," + round4(b[3]));
    }

    placed.embed();
    if (logFile) {
        logFile.writeln("[open] embedded PDF; pageItems=" + doc.pageItems.length +
            ", paths=" + doc.pathItems.length + ", compounds=" + doc.compoundPathItems.length);
    }
    return doc;
}

function openPdfViaSvg(srcFile, logFile) {
    if (typeof system === "undefined" || !system.callSystem) {
        throw new Error("system.callSystem not available for PDF->SVG fallback");
    }

    var airport = airportCodeFromFilename(srcFile.name);
    var svgPath = TMP_DIR + "/" + airport + "_faa_import.svg";
    var cmd = quote(PYTHON_BIN) + " " + quote(PDF_TO_SVG_PY) +
              " --pdf " + quote(srcFile.fsName) +
              " --out " + quote(svgPath) +
              " --page 1 2>&1";
    if (logFile) logFile.writeln("[open] " + cmd);
    var output = system.callSystem(cmd);
    if (logFile) logFile.writeln("[open svg output]\n" + output);

    var svgFile = new File(svgPath);
    if (!svgFile.exists) {
        throw new Error("PDF->SVG fallback produced no SVG. Tail:\n" +
                        (output.length > 500 ? output.substring(output.length - 500) : output));
    }

    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
    var doc = app.open(svgFile, DocumentColorSpace.RGB);
    if (logFile) {
        logFile.writeln("[open] opened SVG; pageItems=" + doc.pageItems.length +
            ", paths=" + doc.pathItems.length + ", compounds=" + doc.compoundPathItems.length);
    }
    return doc;
}

// ================================
// PREPROCESSING
// ================================

function releaseAllClippingMasks(doc, logFile) {
    var changed = true;
    var safety = 30;
    while (changed && safety-- > 0) {
        changed = false;
        var groups = doc.groupItems;
        for (var i = groups.length - 1; i >= 0; i--) {
            if (groups[i].clipped) {
                try {
                    doc.selection = null;
                    groups[i].selected = true;
                    app.executeMenuCommand("releaseMask");
                } catch (e) {}
                changed = true;
                if (logFile) logFile.writeln("  released clipping group; pass " + (30 - safety));
                break;
            }
        }
    }
    try { doc.selection = null; } catch (e) {}
}

function deleteOffArtboardItems(doc) {
    if (doc.artboards.length === 0) return;
    var ab = doc.artboards[0].artboardRect;
    var items = doc.pageItems;
    for (var i = items.length - 1; i >= 0; i--) {
        var p = items[i];
        var b;
        try { b = p.geometricBounds; } catch (e) { continue; }
        if (b[2] < ab[0] || b[0] > ab[2] || b[3] > ab[1] || b[1] < ab[3]) {
            try { p.remove(); } catch (e) {}
        }
    }
}

// ================================
// PATH EXPORT (mirrors ExportClassifiedPaths.jsx)
// ================================

function exportPathsAndEdges(doc, airport, pathsCsvPath, edgesCsvPath) {
    var f = new File(pathsCsvPath);
    f.encoding = "UTF-8";
    f.lineFeed = "Unix";
    f.open("w");
    f.writeln(COLUMNS.join(","));

    var ef = new File(edgesCsvPath);
    ef.encoding = "UTF-8";
    ef.lineFeed = "Unix";
    ef.open("w");
    ef.writeln(EDGE_COLUMNS.join(","));

    var artboard = doc.artboards[0].artboardRect;
    var objectId = 0;

    var compounds = doc.compoundPathItems;
    for (var i = 0; i < compounds.length; i++) {
        var cp = compounds[i];
        var feats = extractFeatures(cp, "compound", airport, objectId, artboard);
        feats.source_layer = "";
        feats.label = "UNLABELED";
        f.writeln(featureRow(feats));
        writeEdges(cp, "compound", airport, objectId, ef);
        objectId++;
    }

    var paths = doc.pathItems;
    for (var j = 0; j < paths.length; j++) {
        var pp = paths[j];
        if (pp.parent && pp.parent.typename === "CompoundPathItem") continue;
        var feats2 = extractFeatures(pp, "path", airport, objectId, artboard);
        feats2.source_layer = "";
        feats2.label = "UNLABELED";
        f.writeln(featureRow(feats2));
        writeEdges(pp, "path", airport, objectId, ef);
        objectId++;
    }

    f.close();
    ef.close();
}

// Below: copies of helpers from ExportClassifiedPaths.jsx so this script
// is self-contained.

function extractFeatures(item, kind, airport, objectId, artboard) {
    var bounds = item.geometricBounds;
    var left = bounds[0], top = bounds[1], right = bounds[2], bottom = bounds[3];
    var width = right - left;
    var height = top - bottom;
    var bboxArea = Math.abs(width * height);
    var aspect = (Math.abs(height) > 1e-9) ? (width / Math.abs(height)) : 0;
    var centroidX = (left + right) / 2;
    var centroidY = (top + bottom) / 2;

    var anchors = 0, subpaths = 0, closed = false;
    var polyArea = -1, perim = 0;
    var allAnchors = [];
    var longestSeg = { angle: 0, length: 0 };

    if (kind === "path") {
        anchors = item.pathPoints.length;
        subpaths = 1;
        closed = item.closed;
        if (closed) polyArea = polyAreaShoelace(item);
        perim = pathPerimeter(item);
        collectAnchors(item, allAnchors);
        updateLongestSegment(item, longestSeg);
    } else {
        var subs = item.pathItems;
        subpaths = subs.length;
        if (subs.length > 0) closed = subs[0].closed;
        for (var k = 0; k < subs.length; k++) {
            anchors += subs[k].pathPoints.length;
            perim += pathPerimeter(subs[k]);
            if (subs[k].closed) {
                if (polyArea < 0) polyArea = 0;
                polyArea += polyAreaShoelace(subs[k]);
            }
            collectAnchors(subs[k], allAnchors);
            updateLongestSegment(subs[k], longestSeg);
        }
    }

    var pca = pcaAngle(allAnchors);
    var filled = !!item.filled;
    var stroked = !!item.stroked;
    var fill = filled ? colorInfo(item.fillColor) : { kind: "none", r: "", g: "", b: "" };
    var stroke = stroked ? colorInfo(item.strokeColor) : { kind: "none", r: "", g: "", b: "" };
    var strokeWidth = stroked ? item.strokeWidth : 0;

    return {
        airport: airport, object_id: objectId, kind: kind,
        source_layer: "", label: "",
        left: round4(left), top: round4(top), right: round4(right), bottom: round4(bottom),
        width: round4(width), height: round4(Math.abs(height)),
        bbox_area: round4(bboxArea), poly_area: round4(polyArea), perimeter: round4(perim),
        centroid_x: round4(centroidX), centroid_y: round4(centroidY),
        aspect: round4(aspect),
        num_anchors: anchors, subpath_count: subpaths, closed: closed ? 1 : 0,
        filled: filled ? 1 : 0,
        fill_kind: fill.kind, fill_r: fill.r, fill_g: fill.g, fill_b: fill.b,
        stroked: stroked ? 1 : 0,
        stroke_kind: stroke.kind, stroke_r: stroke.r, stroke_g: stroke.g, stroke_b: stroke.b,
        stroke_width: round4(strokeWidth),
        principal_angle: round4(pca.angle),
        principal_ratio: round4(pca.ratio),
        longest_segment_angle: round4(longestSeg.angle),
        longest_segment_length: round4(longestSeg.length),
        artboard_left: round4(artboard[0]), artboard_top: round4(artboard[1]),
        artboard_right: round4(artboard[2]), artboard_bottom: round4(artboard[3])
    };
}

function collectAnchors(pathItem, out) {
    var pts = pathItem.pathPoints;
    for (var i = 0; i < pts.length; i++) out.push(pts[i].anchor);
}

function updateLongestSegment(pathItem, current) {
    var pts = pathItem.pathPoints;
    var n = pts.length;
    if (n < 2) return;
    var limit = pathItem.closed ? n : n - 1;
    for (var i = 0; i < limit; i++) {
        var p1 = pts[i].anchor;
        var p2 = pts[(i + 1) % n].anchor;
        var dx = p2[0] - p1[0], dy = p2[1] - p1[1];
        var len = Math.sqrt(dx * dx + dy * dy);
        if (len > current.length) {
            current.length = len;
            var deg = Math.atan2(dy, dx) * 180 / Math.PI;
            current.angle = ((deg % 180) + 180) % 180;
        }
    }
}

function pcaAngle(points) {
    var n = points.length;
    if (n < 2) return { angle: 0, ratio: 1 };
    var mx = 0, my = 0;
    for (var i = 0; i < n; i++) { mx += points[i][0]; my += points[i][1]; }
    mx /= n; my /= n;
    var sxx = 0, syy = 0, sxy = 0;
    for (var j = 0; j < n; j++) {
        var dx = points[j][0] - mx, dy = points[j][1] - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    sxx /= n; syy /= n; sxy /= n;
    var trace = sxx + syy;
    var det = sxx * syy - sxy * sxy;
    var disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
    var lambda1 = trace / 2 + disc;
    var lambda2 = trace / 2 - disc;
    var theta;
    if (Math.abs(sxy) < 1e-12) {
        theta = (sxx >= syy) ? 0 : Math.PI / 2;
    } else {
        theta = Math.atan2(lambda1 - sxx, sxy);
    }
    var deg = theta * 180 / Math.PI;
    deg = ((deg % 180) + 180) % 180;
    var ratio = (lambda2 > 1e-9) ? (lambda1 / lambda2) : 1e6;
    if (ratio > 1e6) ratio = 1e6;
    return { angle: deg, ratio: ratio };
}

function colorInfo(color) {
    if (!color) return { kind: "none", r: "", g: "", b: "" };
    var t = color.typename;
    if (t === "NoColor") return { kind: "none", r: "", g: "", b: "" };
    if (t === "RGBColor") return { kind: "rgb", r: Math.round(color.red), g: Math.round(color.green), b: Math.round(color.blue) };
    if (t === "GrayColor") {
        var v = Math.round(255 - (color.gray * 2.55));
        if (v < 0) v = 0; if (v > 255) v = 255;
        return { kind: "gray", r: v, g: v, b: v };
    }
    if (t === "CMYKColor") {
        var c = color.cyan / 100, m = color.magenta / 100, y = color.yellow / 100, k = color.black / 100;
        return { kind: "cmyk",
                 r: Math.round(255 * (1 - c) * (1 - k)),
                 g: Math.round(255 * (1 - m) * (1 - k)),
                 b: Math.round(255 * (1 - y) * (1 - k)) };
    }
    if (t === "SpotColor" && color.spot && color.spot.color) {
        var inner = colorInfo(color.spot.color);
        return { kind: "spot", r: inner.r, g: inner.g, b: inner.b };
    }
    return { kind: "other", r: "", g: "", b: "" };
}

function polyAreaShoelace(pathItem) {
    var pts = pathItem.pathPoints;
    var n = pts.length;
    if (n < 3) return 0;
    var sum = 0;
    for (var i = 0; i < n; i++) {
        var p1 = pts[i].anchor;
        var p2 = pts[(i + 1) % n].anchor;
        sum += (p1[0] * p2[1]) - (p2[0] * p1[1]);
    }
    return Math.abs(sum) / 2;
}

function pathPerimeter(pathItem) {
    var pts = pathItem.pathPoints;
    var n = pts.length;
    if (n < 2) return 0;
    var sum = 0;
    var limit = pathItem.closed ? n : n - 1;
    for (var i = 0; i < limit; i++) {
        var p1 = pts[i].anchor;
        var p2 = pts[(i + 1) % n].anchor;
        var dx = p2[0] - p1[0], dy = p2[1] - p1[1];
        sum += Math.sqrt(dx * dx + dy * dy);
    }
    return sum;
}

function writeEdges(item, kind, airport, objectId, edgesHandle) {
    if (kind === "path") return writeEdgesOfPath(item, airport, objectId, 0, edgesHandle);
    var subs = item.pathItems;
    for (var k = 0; k < subs.length; k++) writeEdgesOfPath(subs[k], airport, objectId, k, edgesHandle);
}

function writeEdgesOfPath(pathItem, airport, objectId, subpathIndex, edgesHandle) {
    var pts = pathItem.pathPoints;
    var n = pts.length;
    if (n < 2) return;
    var limit = pathItem.closed ? n : n - 1;
    var edgeIndex = 0;
    for (var i = 0; i < limit; i++) {
        var p1 = pts[i].anchor;
        var p2 = pts[(i + 1) % n].anchor;
        var dx = p2[0] - p1[0], dy = p2[1] - p1[1];
        var len = Math.sqrt(dx * dx + dy * dy);
        if (len < 0.001) continue;
        var deg = Math.atan2(dy, dx) * 180 / Math.PI;
        deg = ((deg % 180) + 180) % 180;
        var mx = (p1[0] + p2[0]) / 2;
        var my = (p1[1] + p2[1]) / 2;
        edgesHandle.writeln(
            airport + "," + objectId + "," + subpathIndex + "," + edgeIndex + "," +
            round4(mx) + "," + round4(my) + "," + round4(deg) + "," + round4(len)
        );
        edgeIndex++;
    }
}

function featureRow(f) {
    var out = [];
    for (var i = 0; i < COLUMNS.length; i++) out.push(csvField(f[COLUMNS[i]]));
    return out.join(",");
}
function csvField(v) {
    if (v === undefined || v === null) return "";
    var s = String(v);
    if (s.indexOf(",") >= 0 || s.indexOf("\"") >= 0 || s.indexOf("\n") >= 0) {
        return "\"" + s.replace(/"/g, "\"\"") + "\"";
    }
    return s;
}
function round4(x) {
    if (typeof x !== "number" || !isFinite(x)) return x;
    return Math.round(x * 10000) / 10000;
}

// ================================
// LAYER MANAGEMENT (mirrors ImportPredictedLayers.jsx)
// ================================

function ensureLayers(doc, names) {
    var existing = {};
    for (var i = 0; i < doc.layers.length; i++) existing[doc.layers[i].name] = doc.layers[i];
    var out = {};
    for (var j = 0; j < names.length; j++) {
        var name = names[j];
        if (existing[name]) {
            out[name] = existing[name];
        } else {
            var layer = doc.layers.add();
            layer.name = name;
            out[name] = layer;
        }
    }
    return out;
}

function applyPredictions(doc, idToLabel, layers) {
    var objectId = 0;

    var compounds = doc.compoundPathItems;
    var cpList = [];
    for (var c = 0; c < compounds.length; c++) cpList.push(compounds[c]);
    for (var i = 0; i < cpList.length; i++) {
        var cp = cpList[i];
        var lab = idToLabel[objectId];
        objectId++;
        if (!lab || !layers[lab]) continue;
        try { cp.move(layers[lab], ElementPlacement.PLACEATEND); } catch (e) {}
    }

    var paths = doc.pathItems;
    var pathList = [];
    for (var k = 0; k < paths.length; k++) {
        if (paths[k].parent && paths[k].parent.typename === "CompoundPathItem") continue;
        pathList.push(paths[k]);
    }
    for (var p = 0; p < pathList.length; p++) {
        var pp = pathList[p];
        var lab2 = idToLabel[objectId];
        objectId++;
        if (!lab2 || !layers[lab2]) continue;
        try { pp.move(layers[lab2], ElementPlacement.PLACEATEND); } catch (e) {}
    }
}

// ================================
// JSON + SHELL HELPERS
// ================================

function readClassifyConfig() {
    var f = new File(CONFIG_FILE);
    if (!f.exists) return null;
    return readJson(f);
}

function readJson(file) {
    file.encoding = "UTF-8";
    if (!file.open("r")) return null;
    var text = file.read();
    file.close();
    try {
        if (typeof JSON !== "undefined" && JSON.parse) return JSON.parse(text);
        return (new Function("return (" + text + ")"))();
    } catch (e) {
        return null;
    }
}

function quote(s) { return "'" + String(s).replace(/'/g, "'\\''") + "'"; }

main();
