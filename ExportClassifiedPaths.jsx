/**
 * ExportClassifiedPaths.jsx
 *
 * Walks a folder of <code>-diagram.ai files and emits two CSVs:
 *
 *   <basename>.csv          — one row per vector object (PathItem or
 *                             CompoundPathItem) with geometric features and
 *                             the top-level layer name as label.
 *   <basename>_edges.csv    — one row per anchor-to-anchor edge of every
 *                             polygon, used by relational.py to compute
 *                             local-tangent features (e.g. is this label
 *                             parallel to the nearest taxiway edge).
 *
 * Both files share the same (airport, object_id) primary key.
 *
 * Usage:
 *   File > Scripts > Other Script... > select this file.
 *   Pick the airports root folder (each subfolder named e.g. "atl/" containing
 *   "atl-diagram.ai"). Pick where to save the output CSV (the edges file is
 *   written next to it automatically).
 *
 * One row per object. Children of CompoundPathItems are skipped (the compound
 * itself is the row). PathItems on a layer named "Layer 1" are emitted with
 * label=UNLABELED so the Python loader can drop them.
 */

// ================================
// CONFIGURATION
// ================================

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

// Layer name (case-insensitive) -> training label.
// Accept both spaced and concatenated variants for the label classes.
// The legacy layers (Uncertain, Lines, Text, Arrowheads) and any unrecognized
// layer name are folded into Other — the user's labeling pass left them in
// place and the distinction between them is inconsistent.
function mapLayerToLabel(layerName) {
    var n = (layerName || "").toLowerCase().replace(/\s+/g, " ");
    if (n === "taxiways"   || n === "taxiway")   return "Taxiways";
    if (n === "footprints" || n === "footprint") return "Footprints";
    if (n === "runways"    || n === "runway")    return "Runways";
    if (n === "lights"     || n === "light")     return "Lights";
    if (n === "taxiway labels" || n === "taxiwaylabels" || n === "taxiway label") return "Taxiway Labels";
    if (n === "runway labels"  || n === "runwaylabels"  || n === "runway label")  return "Runway Labels";
    if (n === "stars" || n === "star")           return "Stars";
    if (n === "layer 1" || n === "unclassified") return "UNLABELED";
    // legacy / catch-all → Other
    return "Other";
}

// ================================
// MAIN
// ================================

function main() {
    var rootFolder = Folder.selectDialog("Select airports root folder (contains <code>/<code>-diagram.ai)");
    if (!rootFolder) return;

    var defaultName = "classified_paths.csv";
    var outputFile = File.saveDialog("Save extracted features as CSV", defaultName + ":*.csv");
    if (!outputFile) return;
    if (!/\.csv$/i.test(outputFile.fsName)) {
        outputFile = new File(outputFile.fsName + ".csv");
    }

    var aiFiles = findDiagramFiles(rootFolder);
    if (aiFiles.length === 0) {
        alert("No *-diagram.ai files found under:\n" + rootFolder.fsName);
        return;
    }

    var f = new File(outputFile.fsName);
    f.encoding = "UTF-8";
    if (!f.open("w")) {
        alert("Could not open output file for writing:\n" + outputFile.fsName);
        return;
    }
    f.writeln(COLUMNS.join(","));

    var edgesPath = outputFile.fsName.replace(/\.csv$/i, "") + "_edges.csv";
    var edgesFile = new File(edgesPath);
    edgesFile.encoding = "UTF-8";
    if (!edgesFile.open("w")) {
        f.close();
        alert("Could not open edges file for writing:\n" + edgesPath);
        return;
    }
    edgesFile.writeln(EDGE_COLUMNS.join(","));

    var prevAlerts = app.userInteractionLevel;
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

    var totalObjects = 0;
    var totalEdges = 0;
    var processed = 0;
    var errors = [];
    var startTime = new Date().getTime();

    for (var i = 0; i < aiFiles.length; i++) {
        var aiFile = aiFiles[i];
        var doc = null;
        try {
            doc = app.open(aiFile);
            var airport = airportCodeFromFilename(aiFile.name);
            var counts = exportDocument(doc, airport, f, edgesFile);
            totalObjects += counts.objects;
            totalEdges += counts.edges;
            processed++;
            doc.close(SaveOptions.DONOTSAVECHANGES);
            doc = null;
        } catch (e) {
            errors.push(aiFile.name + ": " + e.message);
            if (doc) {
                try { doc.close(SaveOptions.DONOTSAVECHANGES); } catch (ignore) {}
            }
        }
    }

    f.close();
    edgesFile.close();
    app.userInteractionLevel = prevAlerts;

    var elapsedSec = Math.round((new Date().getTime() - startTime) / 1000);
    var msg = "Export complete.\n\n" +
              "Files processed: " + processed + " / " + aiFiles.length + "\n" +
              "Objects exported: " + totalObjects + "\n" +
              "Edges exported:   " + totalEdges + "\n" +
              "Elapsed: " + elapsedSec + "s\n\n" +
              "CSV:        " + outputFile.fsName + "\n" +
              "Edges CSV:  " + edgesPath;
    if (errors.length > 0) {
        msg += "\n\nErrors (" + errors.length + "):\n" + errors.slice(0, 10).join("\n");
        if (errors.length > 10) msg += "\n... (+" + (errors.length - 10) + " more)";
    }
    alert(msg);
}

// ================================
// FILE DISCOVERY
// ================================

function findDiagramFiles(folder) {
    var results = [];
    var children;
    try { children = folder.getFiles(); } catch (e) { return results; }
    for (var i = 0; i < children.length; i++) {
        var c = children[i];
        if (c instanceof Folder) {
            results = results.concat(findDiagramFiles(c));
        } else if (c instanceof File) {
            if (/-diagram\.ai$/i.test(c.name)) results.push(c);
        }
    }
    return results;
}

function airportCodeFromFilename(name) {
    var m = name.match(/^(.+?)-diagram\.ai$/i);
    return m ? m[1] : name.replace(/\.ai$/i, "");
}

// ================================
// DOCUMENT WALK
// ================================

function exportDocument(doc, airport, fileHandle, edgesHandle) {
    var artboard = doc.artboards[0].artboardRect; // [left, top, right, bottom]
    var objectsWritten = 0;
    var edgesWritten = 0;
    var objectId = 0;

    // Compound paths first — emit one row per compound, skip their children below.
    var compounds = doc.compoundPathItems;
    for (var i = 0; i < compounds.length; i++) {
        var cp = compounds[i];
        var layerName = topLevelLayerName(cp);
        if (!layerName) continue;
        var feats = extractFeatures(cp, "compound", airport, objectId, artboard);
        feats.source_layer = layerName;
        feats.label = mapLayerToLabel(layerName);
        fileHandle.writeln(featureRow(feats));
        edgesWritten += writeEdges(cp, "compound", airport, objectId, edgesHandle);
        objectId++;
        objectsWritten++;
    }

    // PathItems that are not children of a CompoundPathItem.
    var paths = doc.pathItems;
    for (var j = 0; j < paths.length; j++) {
        var p = paths[j];
        if (p.parent && p.parent.typename === "CompoundPathItem") continue;
        var layerName2 = topLevelLayerName(p);
        if (!layerName2) continue;
        var feats2 = extractFeatures(p, "path", airport, objectId, artboard);
        feats2.source_layer = layerName2;
        feats2.label = mapLayerToLabel(layerName2);
        fileHandle.writeln(featureRow(feats2));
        edgesWritten += writeEdges(p, "path", airport, objectId, edgesHandle);
        objectId++;
        objectsWritten++;
    }

    return { objects: objectsWritten, edges: edgesWritten };
}

function topLevelLayerName(item) {
    // Walk up the parent chain; the top-level layer is the Layer whose own
    // parent is the Document. Sublayers walk further up.
    var p = item;
    var safety = 32;
    while (p && p.parent && safety-- > 0) {
        if (p.parent.typename === "Layer" && p.parent.parent && p.parent.parent.typename === "Document") {
            return p.parent.name;
        }
        p = p.parent;
    }
    return null;
}

// ================================
// FEATURE EXTRACTION
// ================================

function extractFeatures(item, kind, airport, objectId, artboard) {
    var bounds = item.geometricBounds; // [left, top, right, bottom] in points; top > bottom
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
        airport: airport,
        object_id: objectId,
        kind: kind,
        source_layer: "",
        label: "",
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
    for (var i = 0; i < pts.length; i++) {
        out.push(pts[i].anchor);
    }
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
            // Normalize to [0, 180): a segment and its reverse share an angle.
            deg = ((deg % 180) + 180) % 180;
            current.angle = deg;
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
        // Illustrator gray is 0..100, where 0 = white and 100 = black
        var v = Math.round(255 - (color.gray * 2.55));
        if (v < 0) v = 0; if (v > 255) v = 255;
        return { kind: "gray", r: v, g: v, b: v };
    }
    if (t === "CMYKColor") {
        var c = color.cyan / 100, m = color.magenta / 100, y = color.yellow / 100, k = color.black / 100;
        var r = Math.round(255 * (1 - c) * (1 - k));
        var g = Math.round(255 * (1 - m) * (1 - k));
        var b = Math.round(255 * (1 - y) * (1 - k));
        return { kind: "cmyk", r: r, g: g, b: b };
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

// Writes one row per anchor-to-anchor edge for the given path or compound path.
// Returns number of edges written. Each subpath of a CompoundPathItem becomes
// its own batch of edges, identified by subpath_index.
function writeEdges(item, kind, airport, objectId, edgesHandle) {
    if (kind === "path") {
        return writeEdgesOfPath(item, airport, objectId, 0, edgesHandle);
    }
    var subs = item.pathItems;
    var written = 0;
    for (var k = 0; k < subs.length; k++) {
        written += writeEdgesOfPath(subs[k], airport, objectId, k, edgesHandle);
    }
    return written;
}

function writeEdgesOfPath(pathItem, airport, objectId, subpathIndex, edgesHandle) {
    var pts = pathItem.pathPoints;
    var n = pts.length;
    if (n < 2) return 0;
    var limit = pathItem.closed ? n : n - 1;
    var written = 0;
    var edgeIndex = 0;
    for (var i = 0; i < limit; i++) {
        var p1 = pts[i].anchor;
        var p2 = pts[(i + 1) % n].anchor;
        var dx = p2[0] - p1[0], dy = p2[1] - p1[1];
        var len = Math.sqrt(dx * dx + dy * dy);
        if (len < 0.001) continue; // skip zero-length artefacts
        var deg = Math.atan2(dy, dx) * 180 / Math.PI;
        deg = ((deg % 180) + 180) % 180; // undirected angle in [0, 180)
        var mx = (p1[0] + p2[0]) / 2;
        var my = (p1[1] + p2[1]) / 2;
        edgesHandle.writeln(
            airport + "," + objectId + "," + subpathIndex + "," + edgeIndex + "," +
            round4(mx) + "," + round4(my) + "," + round4(deg) + "," + round4(len)
        );
        edgeIndex++;
        written++;
    }
    return written;
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

// ================================
// CSV WRITING
// ================================

function featureRow(f) {
    var out = [];
    for (var i = 0; i < COLUMNS.length; i++) {
        out.push(csvField(f[COLUMNS[i]]));
    }
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
main();
