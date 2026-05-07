/**
 * DiagnoseItems.jsx
 *
 * Run this on any document (footprints-wrong.ai, letters-numbers.ai, etc.)
 * to see the Maximum Inscribed Circle (MIC) radius for every filled path.
 *
 * Output is sorted by MIC radius (ascending) so you can see the natural
 * gap between letter strokes (small MIC) and building interiors (large MIC).
 * Set textMaxMICRadiusPts in AirportDiagramClassifier.jsx to a value in that gap.
 *
 * Usage:
 *   1. Open the document in Illustrator
 *   2. File > Scripts > Other Script... → select this file
 *   3. Read the report, find the gap, update CONFIG.textMaxMICRadiusPts
 */

// ── geometry helpers (copied from AirportDiagramClassifier) ──────────────

function collectAllAnchors(item) {
    var pts = [];
    if (item.typename === "PathItem") {
        for (var i = 0; i < item.pathPoints.length; i++)
            pts.push(item.pathPoints[i].anchor);
    } else if (item.typename === "CompoundPathItem") {
        for (var i = 0; i < item.pathItems.length; i++) {
            var sub = item.pathItems[i];
            for (var j = 0; j < sub.pathPoints.length; j++)
                pts.push(sub.pathPoints[j].anchor);
        }
    }
    return pts;
}

function getAnchorCount(item) {
    var n = 0;
    if (item.typename === "PathItem") {
        n = item.pathPoints.length;
    } else if (item.typename === "CompoundPathItem") {
        for (var i = 0; i < item.pathItems.length; i++)
            n += item.pathItems[i].pathPoints.length;
    }
    return n;
}

function buildEdgeList(item) {
    var edges = [];
    function addEdges(pi) {
        var pts = pi.pathPoints;
        if (!pts || pts.length < 2) return;
        for (var i = 0; i < pts.length; i++) {
            var a = pts[i].anchor;
            var b = pts[(i + 1) % pts.length].anchor;
            edges.push([a[0], a[1], b[0], b[1]]);
        }
    }
    if (item.typename === "PathItem") {
        addEdges(item);
    } else if (item.typename === "CompoundPathItem") {
        for (var i = 0; i < item.pathItems.length; i++) addEdges(item.pathItems[i]);
    }
    return edges;
}

function pointInPath(px, py, edges) {
    var crossings = 0;
    for (var i = 0; i < edges.length; i++) {
        var ay = edges[i][1], by = edges[i][3];
        if ((ay <= py && by > py) || (by <= py && ay > py)) {
            var t = (py - ay) / (by - ay);
            var ix = edges[i][0] + t * (edges[i][2] - edges[i][0]);
            if (ix > px) crossings++;
        }
    }
    return (crossings % 2) === 1;
}

function distToSegment(px, py, ax, ay, bx, by) {
    var dx = bx - ax, dy = by - ay;
    var lenSq = dx * dx + dy * dy;
    var t = lenSq < 1e-10 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
    var qx = ax + t * dx - px;
    var qy = ay + t * dy - py;
    return Math.sqrt(qx * qx + qy * qy);
}

function maxInscribedCircle(item, gridSize) {
    var anchors = collectAllAnchors(item);
    if (anchors.length < 3) return 0;

    var minX = anchors[0][0], maxX = anchors[0][0];
    var minY = anchors[0][1], maxY = anchors[0][1];
    for (var i = 1; i < anchors.length; i++) {
        if (anchors[i][0] < minX) minX = anchors[i][0];
        if (anchors[i][0] > maxX) maxX = anchors[i][0];
        if (anchors[i][1] < minY) minY = anchors[i][1];
        if (anchors[i][1] > maxY) maxY = anchors[i][1];
    }

    var edges = buildEdgeList(item);
    if (edges.length === 0) return 0;

    var G = gridSize || 30;
    var stepX = (maxX - minX) / G;
    var stepY = (maxY - minY) / G;
    var maxR = 0;

    for (var gi = 0; gi < G; gi++) {
        for (var gj = 0; gj < G; gj++) {
            var px = minX + (gi + 0.5) * stepX;
            var py = minY + (gj + 0.5) * stepY;
            if (!pointInPath(px, py, edges)) continue;
            var minDist = 1e12;
            for (var e = 0; e < edges.length; e++) {
                var d = distToSegment(px, py, edges[e][0], edges[e][1], edges[e][2], edges[e][3]);
                if (d < minDist) minDist = d;
            }
            if (minDist > maxR) maxR = minDist;
        }
    }
    return maxR;
}

// ── main ─────────────────────────────────────────────────────────────────

var doc = app.activeDocument;
var GRID = 30;        // grid resolution (30×30 is fast; use 50 for more accuracy)
var SQ_PT_PER_SQ_IN = 72 * 72;

var results = [];

for (var i = 0; i < doc.pageItems.length; i++) {
    var item = doc.pageItems[i];

    // Skip compound path sub-paths — they are not independent items
    try {
        if (item.parent && item.parent.typename === "CompoundPathItem") continue;
    } catch (e) {}

    if (item.typename !== "PathItem" && item.typename !== "CompoundPathItem") continue;
    if (!item.filled) continue;

    var anchors = collectAllAnchors(item);
    if (anchors.length < 3) continue;

    var areaIn = Math.abs(item.area) / SQ_PT_PER_SQ_IN;
    var mic    = maxInscribedCircle(item, GRID);
    var nAnch  = getAnchorCount(item);
    var gb     = item.geometricBounds;  // [left, top, right, bottom]

    results.push({
        mic:    mic,
        area:   areaIn,
        anchors: nAnch,
        w:      Math.abs(gb[2] - gb[0]),
        h:      Math.abs(gb[1] - gb[3])
    });
}

// Sort by MIC radius ascending so the gap is obvious
results.sort(function (a, b) { return a.mic - b.mic; });

// Build report
var lines = ["MIC (pt)  Area (sq in)  Anchors  BBox (pt)",
             "--------  ------------  -------  ----------"];
for (var i = 0; i < results.length; i++) {
    var r = results[i];
    var micStr    = ("     " + r.mic.toFixed(2)).slice(-8);
    var areaStr   = ("           " + r.area.toFixed(5)).slice(-12);
    var anchStr   = ("      " + r.anchors).slice(-7);
    var bboxStr   = r.w.toFixed(0) + "×" + r.h.toFixed(0);
    lines.push(micStr + "  " + areaStr + "  " + anchStr + "  " + bboxStr);
}

lines.push("");
lines.push(results.length + " items analysed.");
lines.push("");
lines.push("Find the gap between letter MIC values (small) and");
lines.push("building MIC values (large). Set textMaxMICRadiusPts");
lines.push("in AirportDiagramClassifier.jsx to a value in that gap.");

alert(lines.join("\n"));
