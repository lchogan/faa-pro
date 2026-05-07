/**
 * Airport Diagram Classifier for FAA-PRO
 * Classifies unprocessed airport diagrams into semantic layers
 *
 * Workflow:
 * 1. Release all clipping masks
 * 2. Delete objects outside artboard
 * 3. Classify objects by fill color, stroke, and geometry
 * 4. Move to appropriate layers
 *
 * Layers created:
 * - Taxiways (gray fill)
 * - Footprints (buildings, all sizes)
 * - Runways (large rectangles)
 * - Text (individual letters and labels)
 * - Lines (lat/long lines)
 *
 * Text detection uses the Maximum Inscribed Circle (MIC) approach:
 * The largest circle that fits inside a letter stroke is small (~0.5 pt max).
 * The largest circle that fits inside a building interior is large (1.7 pt min).
 * Since all FAA diagrams share the same font and scale, one absolute MIC
 * threshold (textMaxMICRadiusPts) reliably separates text from buildings.
 *
 * Author: Generated for AOA FAA-PRO
 * Date: 2026-02-19
 */

// ================================
// CONFIGURATION
// ================================
var CONFIG = {
    // === PRE-PROCESSING ===
    releaseClippingMasks: true,
    deleteOffArtboard: true,
    ungroupAll: false,  // Set true if objects are grouped

    // === COLOR DETECTION ===
    // Taxiways are always exactly RGB 207, 207, 207 (#cfcfcf)
    taxiwayRGB: 207,
    taxiwayTolerance: 5,

    // === SIZE THRESHOLDS (in square inches) ===
    textMaxArea: 0.5,             // Individual letters and small labels
    textMinArea: 0.00005,         // Filter tiny artifacts
    footprintMinArea: 0.002,      // Small sheds/buildings
    runwayMinArea: 0.01,          // Minimum area to be considered a runway (sq in)

    // === RUNWAY ASPECT RATIO ===
    // PCA long-axis / short-axis must exceed this to be a runway.
    // Runways are extremely elongated (10:1 to 30:1); nothing else comes close.
    // Even the skinniest letter or building won't reach 5:1 on the principal axis.
    runwayMinAspect: 10,

    // === LINE DETECTION ===
    lineStrokeWidth: 0.39,        // Exact stroke width for lines (points)
    lineStrokeTolerance: 0.05,    // Tolerance for stroke width matching

    // === TEXT DETECTION (Maximum Inscribed Circle) ===
    // Letters have thin strokes bounded by the font's stroke width.
    // All FAA diagrams share the same font/scale, so one absolute MIC radius
    // threshold is simpler and more reliable than a scale-invariant ratio.
    // Grid resolution for MIC sampling (higher = slower but more accurate)
    micGridSize: 50,
    // Maximum inscribed circle radius (in document points) for a character stroke.
    // Calibrated from DiagnoseItems.jsx data:
    //   Letters (letters-numbers.ai): max MIC = 0.50pt
    //   Buildings (footprints-wrong.ai): min MIC = 1.70pt
    // 1.0pt sits in the middle of the 3.4× gap.
    textMaxMICRadiusPts: 1.0,

    // === ARROWHEAD DETECTION ===
    // Arrowheads have one very acute tip. Any filled shape whose sharpest
    // interior vertex angle is below this threshold is classified as an arrowhead.
    arrowheadMaxAngle: 30,   // degrees

    // === OUTPUT LAYERS ===
    layerTaxiways: "Taxiways",
    layerFootprints: "Footprints",
    layerRunways: "Runways",
    layerLights: "Lights",       // blank placeholder, sits just above Runways
    layerArrowheads: "Arrowheads",
    layerText: "Text",
    layerLines: "Lines",
    layerUncertain: "Uncertain",

    // === DEBUG ===
    debugMode: true,
    showReport: true,
    logDetails: false  // Set true for verbose logging
};

// ================================
// UTILITY FUNCTIONS
// ================================

function log(msg) {
    if (CONFIG.logDetails) {
        $.writeln(msg);
    }
}

function pointsToInches(points) {
    return points / 72;
}

function sqPointsToSqInches(sqPoints) {
    return sqPoints / (72 * 72);
}

// ================================
// GEOMETRY FUNCTIONS
// ================================

function getItemBounds(item) {
    var bounds = item.geometricBounds; // [left, top, right, bottom]
    var width = Math.abs(bounds[2] - bounds[0]);
    var height = Math.abs(bounds[1] - bounds[3]);
    return {
        left: bounds[0],
        top: bounds[1],
        right: bounds[2],
        bottom: bounds[3],
        width: width,
        height: height,
        centerX: bounds[0] + width / 2,
        centerY: bounds[3] + height / 2
    };
}

function isInArtboard(item, artboard) {
    var bounds = getItemBounds(item);
    var ab = artboard.artboardRect; // [left, top, right, bottom]

    // Check if item center is within artboard
    var centerInside = (
        bounds.centerX >= ab[0] &&
        bounds.centerX <= ab[2] &&
        bounds.centerY <= ab[1] &&
        bounds.centerY >= ab[3]
    );

    return centerInside;
}

function getAnchorCount(item) {
    var total = 0;
    if (item.typename === "PathItem") {
        total = item.pathPoints.length;
    } else if (item.typename === "CompoundPathItem") {
        for (var i = 0; i < item.pathItems.length; i++) {
            total += item.pathItems[i].pathPoints.length;
        }
    }
    return total;
}

// ── principal-axis bounding box ────────────────────────────────────────────
//
// Uses PCA on the anchor points to find the natural orientation of the shape,
// independent of page rotation. Returns { w, h } where w >= h.

function collectAllAnchors(item) {
    var pts = [];
    if (item.typename === "PathItem") {
        for (var i = 0; i < item.pathPoints.length; i++) {
            pts.push(item.pathPoints[i].anchor);
        }
    } else if (item.typename === "CompoundPathItem") {
        for (var i = 0; i < item.pathItems.length; i++) {
            var sub = item.pathItems[i];
            for (var j = 0; j < sub.pathPoints.length; j++) {
                pts.push(sub.pathPoints[j].anchor);
            }
        }
    }
    return pts;
}

function getPrincipalBounds(anchors) {
    var n = anchors.length;
    if (n < 2) return { w: 0, h: 0 };

    // centroid
    var cx = 0, cy = 0;
    for (var i = 0; i < n; i++) { cx += anchors[i][0]; cy += anchors[i][1]; }
    cx /= n; cy /= n;

    // covariance
    var mxx = 0, myy = 0, mxy = 0;
    for (var i = 0; i < n; i++) {
        var dx = anchors[i][0] - cx;
        var dy = anchors[i][1] - cy;
        mxx += dx * dx;
        myy += dy * dy;
        mxy += dx * dy;
    }

    // principal axis angle
    var angle = 0.5 * Math.atan2(2 * mxy, mxx - myy);
    var cosA  = Math.cos(angle);
    var sinA  = Math.sin(angle);

    // project and measure extents
    var minU =  1e12, maxU = -1e12;
    var minV =  1e12, maxV = -1e12;
    for (var i = 0; i < n; i++) {
        var dx = anchors[i][0] - cx;
        var dy = anchors[i][1] - cy;
        var u =  dx * cosA + dy * sinA;
        var v = -dx * sinA + dy * cosA;
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
    }

    var a = maxU - minU;
    var b = maxV - minV;
    return { w: Math.max(a, b), h: Math.min(a, b) };
}

// ================================
// MAXIMUM INSCRIBED CIRCLE (MIC)
// ================================
//
// Letters have thin strokes; the largest circle that fits inside a letter is
// bounded by the stroke width (calibrated max: 0.50 pt). Buildings, even
// complex ones, have wide interior sections (calibrated min: 1.70 pt).
// textMaxMICRadiusPts = 1.0 pt sits in the middle of that gap.
//
// Implementation: sample a grid of points inside the ink area (using even-odd
// ray casting through all boundary edges), then for each interior sample find
// the distance to the nearest edge segment. The maximum of those distances is
// the approximate MIC radius.

/**
 * Build a flat list of line segments [ax, ay, bx, by] from all sub-paths.
 * Each sub-path is closed (last point connects back to first).
 */
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

/**
 * Even-odd ray casting: true if (px, py) is inside the ink area.
 * Counts horizontal-rightward crossings through all edges.
 * Works correctly for compound paths with holes.
 */
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

/**
 * Shortest distance from point (px, py) to line segment (ax,ay)→(bx,by).
 */
function distToSegment(px, py, ax, ay, bx, by) {
    var dx = bx - ax, dy = by - ay;
    var lenSq = dx * dx + dy * dy;
    var t = lenSq < 1e-10 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
    var qx = ax + t * dx - px;
    var qy = ay + t * dy - py;
    return Math.sqrt(qx * qx + qy * qy);
}

/**
 * Approximate maximum inscribed circle radius via grid sampling.
 * Returns radius in document points.
 */
function maxInscribedCircle(item) {
    var anchors = collectAllAnchors(item);
    if (anchors.length < 3) return 0;

    // Axis-aligned bounding box for the sampling grid
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

    var G = CONFIG.micGridSize;
    var stepX = (maxX - minX) / G;
    var stepY = (maxY - minY) / G;
    var maxR = 0;

    for (var gi = 0; gi < G; gi++) {
        for (var gj = 0; gj < G; gj++) {
            var px = minX + (gi + 0.5) * stepX;
            var py = minY + (gj + 0.5) * stepY;

            if (!pointInPath(px, py, edges)) continue;

            // Minimum distance from this interior point to any boundary edge
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

// ================================
// MARGIN BOX DETECTION
// ================================

/**
 * Find the large unfilled stroked rectangle that borders the diagram.
 * Criteria: PathItem, no fill, has stroke, ≤8 anchors, bounding-box area
 * ≥ 100,000 sq pts (~19 sq in). Returns the largest qualifying item's bounds,
 * or null if none found.
 */
function findMarginBox(doc) {
    var best     = null;
    var bestArea = 0;

    for (var i = 0; i < doc.pageItems.length; i++) {
        var item = doc.pageItems[i];
        if (item.typename !== "PathItem") continue;
        if (item.filled)   continue;   // must have no fill
        if (!item.stroked) continue;   // must have a stroke
        if (item.pathPoints.length > 8) continue;  // simple rectangle

        var b = getItemBounds(item);
        var bboxArea = b.width * b.height;  // sq points

        if (bboxArea < 100000) continue;   // ~19 sq inches minimum

        if (bboxArea > bestArea) {
            bestArea = bboxArea;
            best = b;
        }
    }
    return best;  // { left, top, right, bottom, width, height, centerX, centerY }
}

/**
 * True if the item's centre point lies outside the margin box.
 * Uses centre so objects that straddle the border stay inside.
 */
function isOutsideMarginBox(itemBounds, box) {
    return (
        itemBounds.centerX < box.left   ||
        itemBounds.centerX > box.right  ||
        itemBounds.centerY < box.bottom ||
        itemBounds.centerY > box.top
    );
}

/**
 * True if the item's bounds match the margin box within 5 pts on each edge.
 * Used to skip the border rectangle itself — don't move it to any layer.
 */
function isMarginBoxItem(itemBounds, box) {
    var tol = 5;
    return (
        Math.abs(itemBounds.left   - box.left)   < tol &&
        Math.abs(itemBounds.right  - box.right)  < tol &&
        Math.abs(itemBounds.top    - box.top)    < tol &&
        Math.abs(itemBounds.bottom - box.bottom) < tol
    );
}

// ================================
// COLOR DETECTION
// ================================

function colorToRGB(color) {
    if (!color) return null;
    if (color.typename === "RGBColor") {
        return {
            r: Math.round(color.red),
            g: Math.round(color.green),
            b: Math.round(color.blue)
        };
    } else if (color.typename === "CMYKColor") {
        var r = Math.round(255 * (1 - color.cyan / 100) * (1 - color.black / 100));
        var g = Math.round(255 * (1 - color.magenta / 100) * (1 - color.black / 100));
        var b = Math.round(255 * (1 - color.yellow / 100) * (1 - color.black / 100));
        return { r: r, g: g, b: b };
    } else if (color.typename === "GrayColor") {
        var val = Math.round(255 * (100 - color.gray) / 100);
        return { r: val, g: val, b: val };
    } else if (color.typename === "SpotColor") {
        // SpotColor is a named swatch; unwrap the underlying color and apply tint.
        // tint=100 → full color, tint=0 → white.
        var base = colorToRGB(color.spot.color);
        if (!base) return null;
        var t = (color.tint != null ? color.tint : 100) / 100;
        return {
            r: Math.round(255 + (base.r - 255) * t),
            g: Math.round(255 + (base.g - 255) * t),
            b: Math.round(255 + (base.b - 255) * t)
        };
    }
    return null;
}

function getRGBColor(item) {
    if (!item.filled || !item.fillColor) return null;
    return colorToRGB(item.fillColor);
}

function isWhiteFill(item) {
    var color = item.fillColor;
    if (!color) return false;
    if (color.typename === "RGBColor") {
        return color.red >= 250 && color.green >= 250 && color.blue >= 250;
    }
    if (color.typename === "CMYKColor") {
        return color.cyan <= 2 && color.magenta <= 2 && color.yellow <= 2 && color.black <= 2;
    }
    if (color.typename === "GrayColor") {
        return color.gray <= 2;  // GrayColor.gray=0 is white in Illustrator
    }
    return false;
}

function isGrayFill(item) {
    var rgb = getRGBColor(item);
    if (!rgb) return false;
    var t = CONFIG.taxiwayTolerance;
    var v = CONFIG.taxiwayRGB;
    return Math.abs(rgb.r - v) <= t && Math.abs(rgb.g - v) <= t && Math.abs(rgb.b - v) <= t;
}

// Returns the minimum interior angle (degrees) at any anchor vertex of the item.
// Used to detect arrowheads, which have one very acute tip.
function minVertexAngle(item) {
    var anchors = collectAllAnchors(item);
    var n = anchors.length;
    if (n < 3) return 360;
    var minAngle = 360;
    for (var i = 0; i < n; i++) {
        var prev = anchors[(i - 1 + n) % n];
        var curr = anchors[i];
        var next = anchors[(i + 1) % n];
        var ax = prev[0] - curr[0], ay = prev[1] - curr[1];
        var bx = next[0] - curr[0], by = next[1] - curr[1];
        var lenA = Math.sqrt(ax * ax + ay * ay);
        var lenB = Math.sqrt(bx * bx + by * by);
        if (lenA < 1e-6 || lenB < 1e-6) continue;
        var cosA = (ax * bx + ay * by) / (lenA * lenB);
        cosA = Math.max(-1, Math.min(1, cosA));
        var angle = Math.acos(cosA) * 180 / Math.PI;
        if (angle < minAngle) minAngle = angle;
    }
    return minAngle;
}


// ================================
// CLASSIFICATION
// ================================

function classifyItem(item) {
    // Only process paths and compound paths
    if (item.typename !== "PathItem" && item.typename !== "CompoundPathItem") {
        return { type: "skip", reason: "Not a path" };
    }

    // Get basic metrics
    var bounds = getItemBounds(item);
    var widthIn = pointsToInches(bounds.width);
    var heightIn = pointsToInches(bounds.height);
    var areaIn = sqPointsToSqInches(Math.abs(item.area));
    var aspectRatio = bounds.height > 0 ? bounds.width / bounds.height : 0;
    var anchorCount = getAnchorCount(item);

    var metrics = {
        width: widthIn,
        height: heightIn,
        area: areaIn,
        aspect: aspectRatio,
        anchors: anchorCount,
        hasFill: item.filled,
        hasStroke: item.stroked
    };

    log("Analyzing: " + widthIn.toFixed(4) + "\" × " + heightIn.toFixed(4) + "\", area: " + areaIn.toFixed(6) + " sq in, anchors: " + anchorCount);

    // === LINES (FIRST PRIORITY - 0.39pt stroke) ===
    // Check this BEFORE anything else
    if (item.stroked) {
        var strokeWidth = item.strokeWidth;
        var isCorrectStroke = Math.abs(strokeWidth - CONFIG.lineStrokeWidth) <= CONFIG.lineStrokeTolerance;

        if (isCorrectStroke) {
            // This is a line - classify immediately, don't check anything else
            return {
                type: "line",
                reason: "0.39pt stroke width (line)",
                metrics: metrics,
                strokeWidth: strokeWidth
            };
        }
    }

    // === TAXIWAYS (gray fill) ===
    if (item.filled && isGrayFill(item)) {
        return {
            type: "taxiway",
            reason: "Gray fill color detected",
            metrics: metrics
        };
    }

    // === WHITE FILL (character counter shapes) ===
    // Letters like O, B, D, 0, 8 etc. are stored as two stacked objects:
    // an outer black shape + a separate inner white-filled shape. The white
    // shape must go to the Text layer alongside its outer letter so the
    // visual hole effect is preserved.
    if (item.filled && isWhiteFill(item) && areaIn <= CONFIG.textMaxArea) {
        return {
            type: "text",
            reason: "White fill (character counter shape)",
            metrics: metrics
        };
    }

    // === TEXT (Maximum Inscribed Circle absolute radius test) ===
    // Letters have thin strokes bounded by the font's stroke width.
    // Since all FAA diagrams share the same font and scale, one absolute MIC
    // radius threshold is all that's needed — no ratio or long-axis cap required.
    // Buildings (even thin ones) have wider interior sections than letter strokes.
    if (item.filled && areaIn >= CONFIG.textMinArea && areaIn <= CONFIG.textMaxArea) {
        var micRadius = maxInscribedCircle(item);
        if (micRadius < CONFIG.textMaxMICRadiusPts) {
            return {
                type: "text",
                reason: "Thin strokes (MIC " + micRadius.toFixed(1) + "pt)",
                metrics: metrics
            };
        }
        // MIC too large — wide interior sections, continue to shape checks.
    }

    // === RUNWAYS ===
    // Runways are uniquely elongated — PCA aspect ratios of 10:1 to 30:1.
    // No letter or building comes close to 5:1 on the principal axis.
    if (item.filled && areaIn >= CONFIG.runwayMinArea) {
        var rwAnchors = collectAllAnchors(item);
        var rwPB      = getPrincipalBounds(rwAnchors);
        var paAspect  = rwPB.h > 0 ? rwPB.w / rwPB.h : 0;  // always >= 1

        if (paAspect >= CONFIG.runwayMinAspect) {
            return {
                type: "runway",
                reason: "Runway: " + areaIn.toFixed(2) + " sq in, aspect " + paAspect.toFixed(1) + ":1",
                metrics: metrics
            };
        }
    }

    // === SYMBOL CIRCLES → UNCERTAIN ===
    // FAA obstruction/compass marker circles are always exactly 3.67×3.67 pt.
    // They don't belong in Footprints.
    if (item.filled && Math.abs(bounds.width - 3.67) < 0.5 && Math.abs(bounds.height - 3.67) < 0.5) {
        return {
            type: "uncertain",
            reason: "Symbol circle 3.67pt (review manually)",
            metrics: metrics
        };
    }

    // === ARROWHEADS ===
    // Arrowheads have one very acute tip vertex. Check before footprints so they
    // don't fall through to the buildings layer.
    if (item.filled) {
        var tipAngle = minVertexAngle(item);
        if (tipAngle < CONFIG.arrowheadMaxAngle) {
            return {
                type: "arrowhead",
                reason: "Arrowhead: tip angle " + tipAngle.toFixed(1) + "°",
                metrics: metrics
            };
        }
    }

    // === FOOTPRINTS (buildings - filled objects large enough to be structures) ===
    if (item.filled && areaIn >= CONFIG.footprintMinArea) {
        return {
            type: "footprint",
            reason: "Building footprint: " + areaIn.toFixed(4) + " sq in",
            metrics: metrics
        };
    }

    // === UNCERTAIN ===
    return {
        type: "uncertain",
        reason: "Doesn't match any category (area: " + areaIn.toFixed(6) + " sq in, filled: " + item.filled + ", stroked: " + item.stroked + ")",
        metrics: metrics
    };
}

// ================================
// PRE-PROCESSING
// ================================

function releaseAllClippingMasks(doc) {
    var count = 0;
    // Release clipping masks
    for (var i = doc.pageItems.length - 1; i >= 0; i--) {
        try {
            var item = doc.pageItems[i];
            if (item.typename === "GroupItem" && item.clipped) {
                item.clipped = false;
                count++;
            }
        } catch (e) {
            // Continue on error
        }
    }
    return count;
}

function deleteOffArtboardItems(doc) {
    var artboard = doc.artboards[0];
    var count = 0;

    for (var i = doc.pageItems.length - 1; i >= 0; i--) {
        try {
            var item = doc.pageItems[i];
            if (!isInArtboard(item, artboard)) {
                item.remove();
                count++;
            }
        } catch (e) {
            // Continue on error
        }
    }
    return count;
}

function ungroupAllItems(doc) {
    var count = 0;
    var maxIterations = 100;  // Prevent infinite loop
    var iteration = 0;

    while (iteration < maxIterations) {
        var foundGroup = false;
        for (var i = doc.pageItems.length - 1; i >= 0; i--) {
            try {
                var item = doc.pageItems[i];
                if (item.typename === "GroupItem") {
                    // Move all children to parent
                    while (item.pageItems.length > 0) {
                        item.pageItems[0].move(doc, ElementPlacement.PLACEATEND);
                    }
                    item.remove();
                    count++;
                    foundGroup = true;
                }
            } catch (e) {
                // Continue on error
            }
        }

        if (!foundGroup) break;
        iteration++;
    }

    return count;
}

// ================================
// LAYER MANAGEMENT
// ================================

function getOrCreateLayer(doc, layerName) {
    try {
        return doc.layers.getByName(layerName);
    } catch (e) {
        var layer = doc.layers.add();
        layer.name = layerName;
        return layer;
    }
}

// ================================
// MAIN PROCESSING
// ================================

function processDocument() {
    var doc = app.activeDocument;

    var stats = {
        totalProcessed: 0,
        clippingMasksReleased: 0,
        offArtboardDeleted: 0,
        groupsUngrouped: 0,
        taxiways: 0,
        footprints: 0,
        runways: 0,
        arrowheads: 0,
        text: 0,
        lines: 0,
        uncertain: 0,
        skipped: 0
    };

    // === PRE-PROCESSING ===
    if (CONFIG.releaseClippingMasks) {
        stats.clippingMasksReleased = releaseAllClippingMasks(doc);
    }

    if (CONFIG.ungroupAll) {
        stats.groupsUngrouped = ungroupAllItems(doc);
    }

    if (CONFIG.deleteOffArtboard) {
        stats.offArtboardDeleted = deleteOffArtboardItems(doc);
    }

    // === CREATE LAYERS ===
    var layers = {
        taxiways: getOrCreateLayer(doc, CONFIG.layerTaxiways),
        footprints: getOrCreateLayer(doc, CONFIG.layerFootprints),
        runways: getOrCreateLayer(doc, CONFIG.layerRunways),
        lights: getOrCreateLayer(doc, CONFIG.layerLights),
        arrowheads: getOrCreateLayer(doc, CONFIG.layerArrowheads),
        text: getOrCreateLayer(doc, CONFIG.layerText),
        lines: getOrCreateLayer(doc, CONFIG.layerLines),
        uncertain: getOrCreateLayer(doc, CONFIG.layerUncertain)
    };

    // === FIND DIAGRAM MARGIN BOX ===
    // The large unfilled stroked rectangle that borders the diagram.
    // Everything outside it is a text label/annotation.
    var marginBox = findMarginBox(doc);

    // === CLASSIFY ITEMS ===
    var allItems = [];
    var outsideCount = 0;

    for (var i = 0; i < doc.pageItems.length; i++) {
        var item = doc.pageItems[i];

        // doc.pageItems is a deep/flat collection that includes the sub-paths
        // of CompoundPathItems as separate entries. If we process them
        // individually they get moved out of the compound path, destroying the
        // hole structure (the sub-path lands on the Footprints layer while the
        // outer path goes to Text — leaving a solid letter with no hole).
        // The CompoundPathItem itself is also in the list and will be moved as
        // a single unit, carrying all its sub-paths with it.
        try {
            if (item.parent && item.parent.typename === "CompoundPathItem") continue;
        } catch (e) {}

        var ib   = getItemBounds(item);

        // Skip the border rectangle itself — leave it in place.
        if (marginBox && isMarginBoxItem(ib, marginBox)) continue;

        // Everything outside the diagram border is a text label/annotation.
        if (marginBox && isOutsideMarginBox(ib, marginBox)) {
            allItems.push({
                item: item,
                classification: {
                    type: "text",
                    reason: "Outside diagram margin box",
                    metrics: null
                }
            });
            outsideCount++;
            continue;
        }

        allItems.push({
            item: item,
            classification: classifyItem(item)
        });
    }

    // === MOVE ITEMS TO LAYERS ===
    for (var i = 0; i < allItems.length; i++) {
        var obj = allItems[i];
        var item = obj.item;
        var cls = obj.classification;

        stats.totalProcessed++;

        // Add debug info
        if (CONFIG.debugMode && cls.metrics) {
            item.note = "Type: " + cls.type + "\n" +
                       "Reason: " + cls.reason + "\n" +
                       "Size: " + cls.metrics.width.toFixed(4) + "\" × " + cls.metrics.height.toFixed(4) + "\"\n" +
                       "Area: " + cls.metrics.area.toFixed(6) + " sq in\n" +
                       "Aspect: " + cls.metrics.aspect.toFixed(2) + "\n" +
                       "Anchors: " + cls.metrics.anchors;
        }

        // Determine target layer
        var targetLayer = layers.uncertain;

        switch (cls.type) {
            case "taxiway":
                targetLayer = layers.taxiways;
                stats.taxiways++;
                break;
            case "footprint":
                targetLayer = layers.footprints;
                stats.footprints++;
                break;
            case "runway":
                targetLayer = layers.runways;
                stats.runways++;
                break;
            case "arrowhead":
                targetLayer = layers.arrowheads;
                stats.arrowheads++;
                break;
            case "text":
                targetLayer = layers.text;
                stats.text++;
                break;
            case "line":
            case "line_other":
                targetLayer = layers.lines;
                stats.lines++;
                break;
            case "skip":
                stats.skipped++;
                continue;
            default:
                stats.uncertain++;
        }

        // Move to layer
        try {
            item.move(targetLayer, ElementPlacement.PLACEATEND);
            // Compound paths need even-odd fill rule so their holes render correctly
            // after being moved. Non-zero winding can fill holes if sub-path
            // directions are not perfectly reversed.
            if (item.typename === "CompoundPathItem") {
                try { item.evenodd = true; } catch (e2) {}
            }
        } catch (e) {
            log("Error moving item: " + e);
        }
    }

    // === POST-PROCESSING ===

    // 1. Pathfinder Unite — merge all taxiway paths into one shape
    if (stats.taxiways > 1) {
        try {
            layers.taxiways.locked  = false;
            layers.taxiways.visible = true;
            doc.activeLayer = layers.taxiways;
            doc.selection   = null;
            for (var ti = 0; ti < layers.taxiways.pageItems.length; ti++) {
                layers.taxiways.pageItems[ti].selected = true;
            }
            app.executeMenuCommand("group");          // group the selection
            app.executeMenuCommand("Live Pathfinder Add");  // Effect > Pathfinder > Add
            app.executeMenuCommand("expandStyle");    // Object > Expand Appearance
            doc.selection = null;
        } catch (e) {
            log("Taxiway unite failed: " + e);
        }
    }

    // 2. Delete any empty layers except Lights (which is an intentional blank placeholder)
    var keepNames = {};
    for (var kn in CONFIG) {
        if (kn.indexOf("layer") === 0) keepNames[CONFIG[kn]] = true;
    }
    for (var di = doc.layers.length - 1; di >= 0; di--) {
        var dl = doc.layers[di];
        if (dl.pageItems.length === 0 && dl.name !== CONFIG.layerLights) {
            try { dl.locked = false; dl.remove(); } catch (e) {}
        }
    }

    // 3. Reorder — Taxiways at the very bottom, Footprints above, Runways above,
    //    then Lights immediately above Runways
    try {
        layers.taxiways.move(doc, ElementPlacement.PLACEATEND);
        layers.footprints.move(layers.taxiways, ElementPlacement.PLACEBEFORE);
        layers.runways.move(layers.footprints, ElementPlacement.PLACEBEFORE);
        layers.lights.move(layers.runways, ElementPlacement.PLACEBEFORE);
    } catch (e) {
        log("Layer reorder failed: " + e);
    }

    // 4. Hide all layers
    for (var li = 0; li < doc.layers.length; li++) {
        doc.layers[li].visible = false;
    }

    // === REPORT ===
    if (CONFIG.showReport) {
        var report = "=== AIRPORT DIAGRAM CLASSIFICATION COMPLETE ===\n\n";

        report += "PRE-PROCESSING:\n";
        report += "  Clipping masks released: " + stats.clippingMasksReleased + "\n";
        report += "  Groups ungrouped: " + stats.groupsUngrouped + "\n";
        report += "  Off-artboard deleted: " + stats.offArtboardDeleted + "\n\n";

        report += "MARGIN BOX:\n";
        if (marginBox) {
            report += "  Found: " + marginBox.width.toFixed(1) + " × " + marginBox.height.toFixed(1) + " pts\n";
            report += "  Items outside (→ Text): " + outsideCount + "\n\n";
        } else {
            report += "  NOT FOUND — all items classified normally\n\n";
        }

        report += "CLASSIFICATION:\n";
        report += "  Total processed: " + stats.totalProcessed + "\n";
        report += "  Taxiways: " + stats.taxiways + "\n";
        report += "  Footprints: " + stats.footprints + "\n";
        report += "  Runways: " + stats.runways + "\n";
        report += "  Arrowheads: " + stats.arrowheads + "\n";
        report += "  Text: " + stats.text + "\n";
        report += "  Lines: " + stats.lines + "\n";
        report += "  Uncertain: " + stats.uncertain + "\n";
        report += "  Skipped: " + stats.skipped + "\n\n";

        report += "Debug mode: " + (CONFIG.debugMode ? "ON (check object notes)" : "OFF");

        alert(report);
    }
}

// ================================
// ENTRY POINT
// ================================

try {
    if (app.documents.length === 0) {
        alert("Please open an airport diagram document first.");
    } else {
        processDocument();
    }
} catch (err) {
    alert("Error: " + err.message + "\nLine: " + err.line);
}
