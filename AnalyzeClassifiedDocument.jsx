/**
 * Analyze Classified Document
 * Extracts metrics from an already-classified document to build optimal thresholds
 *
 * Instructions:
 * 1. Open your classified cha-diagram.ai in Illustrator
 * 2. Run this script
 * 3. It will create a CSV file on your Desktop with all metrics
 */

// ================================
// UTILITY FUNCTIONS (same as main script)
// ================================

function pointsToInches(points) {
    return points / 72;
}

function sqPointsToSqInches(sqPoints) {
    return sqPoints / (72 * 72);
}

function calculatePerimeter(pathItem) {
    if (!pathItem || !pathItem.pathPoints) return 0;
    var perimeter = 0;
    var points = pathItem.pathPoints;
    for (var i = 0; i < points.length; i++) {
        var p1 = points[i].anchor;
        var p2 = points[(i + 1) % points.length].anchor;
        var dx = p2[0] - p1[0];
        var dy = p2[1] - p1[1];
        perimeter += Math.sqrt(dx * dx + dy * dy);
    }
    return perimeter;
}

function countHoles(item) {
    if (item.typename === "CompoundPathItem") {
        return Math.max(0, item.pathItems.length - 1);
    }
    return 0;
}

function getGrossArea(item) {
    if (item.typename === "CompoundPathItem" && item.pathItems.length > 0) {
        var maxArea = 0;
        for (var i = 0; i < item.pathItems.length; i++) {
            var subArea = Math.abs(item.pathItems[i].area);
            if (subArea > maxArea) {
                maxArea = subArea;
            }
        }
        return maxArea;
    }
    return Math.abs(item.area);
}

function getTotalPerimeter(item) {
    var totalPerim = 0;
    if (item.typename === "CompoundPathItem") {
        for (var i = 0; i < item.pathItems.length; i++) {
            totalPerim += calculatePerimeter(item.pathItems[i]);
        }
    } else if (item.typename === "PathItem") {
        totalPerim = calculatePerimeter(item);
    }
    return totalPerim;
}

function getTotalAnchors(item) {
    var totalAnchors = 0;
    if (item.typename === "CompoundPathItem") {
        for (var i = 0; i < item.pathItems.length; i++) {
            totalAnchors += item.pathItems[i].pathPoints.length;
        }
    } else if (item.typename === "PathItem") {
        totalAnchors = item.pathPoints.length;
    }
    return totalAnchors;
}

function analyzeItem(item) {
    var metrics = {};
    var bounds = item.geometricBounds;
    var width = Math.abs(bounds[2] - bounds[0]);
    var height = Math.abs(bounds[1] - bounds[3]);

    metrics.width = pointsToInches(width);
    metrics.height = pointsToInches(height);
    metrics.aspectRatio = height > 0 ? width / height : 0;
    metrics.netArea = sqPointsToSqInches(Math.abs(item.area));
    metrics.grossArea = sqPointsToSqInches(getGrossArea(item));

    var boundingArea = metrics.width * metrics.height;
    metrics.occupancyRatio = boundingArea > 0 ? metrics.netArea / boundingArea : 0;
    metrics.grossOccupancyRatio = boundingArea > 0 ? metrics.grossArea / boundingArea : 0;

    metrics.holeCount = countHoles(item);
    metrics.perimeter = pointsToInches(getTotalPerimeter(item));
    metrics.anchorCount = getTotalAnchors(item);
    metrics.anchorDensity = metrics.perimeter > 0 ? metrics.anchorCount / metrics.perimeter : 0;
    metrics.perimeterAreaRatio = metrics.netArea > 0 ? metrics.perimeter / metrics.netArea : 0;

    // Get fill and stroke properties
    metrics.hasFill = item.filled;
    metrics.hasStroke = item.stroked;
    metrics.fillColor = "none";
    metrics.strokeWidth = item.stroked ? pointsToInches(item.strokeWidth) : 0;

    if (item.filled && item.fillColor) {
        if (item.fillColor.typename === "RGBColor") {
            metrics.fillColor = "RGB(" +
                Math.round(item.fillColor.red) + "," +
                Math.round(item.fillColor.green) + "," +
                Math.round(item.fillColor.blue) + ")";
        } else if (item.fillColor.typename === "CMYKColor") {
            metrics.fillColor = "CMYK(" +
                Math.round(item.fillColor.cyan) + "," +
                Math.round(item.fillColor.magenta) + "," +
                Math.round(item.fillColor.yellow) + "," +
                Math.round(item.fillColor.black) + ")";
        } else if (item.fillColor.typename === "GrayColor") {
            metrics.fillColor = "Gray(" + Math.round(item.fillColor.gray) + ")";
        }
    }

    return metrics;
}

// ================================
// MAIN ANALYSIS
// ================================

function analyzeDocument() {
    var doc = app.activeDocument;

    // CSV header
    var csv = "Layer,ObjectType,Width_in,Height_in,NetArea_sqin,GrossArea_sqin,Occupancy,GrossOccupancy,Holes,Perimeter_in,Anchors,AnchorDensity,PerimAreaRatio,AspectRatio,HasFill,HasStroke,FillColor,StrokeWidth_in\n";

    var stats = {
        totalObjects: 0,
        byLayer: {}
    };

    // Iterate through all layers
    for (var i = 0; i < doc.layers.length; i++) {
        var layer = doc.layers[i];
        var layerName = layer.name;

        if (!stats.byLayer[layerName]) {
            stats.byLayer[layerName] = 0;
        }

        // Process all page items in this layer
        for (var j = 0; j < layer.pageItems.length; j++) {
            var item = layer.pageItems[j];

            // Only analyze paths and compound paths
            if (item.typename !== "PathItem" && item.typename !== "CompoundPathItem") {
                continue;
            }

            var metrics = analyzeItem(item);
            stats.totalObjects++;
            stats.byLayer[layerName]++;

            // Add to CSV
            csv += layerName + ",";
            csv += item.typename + ",";
            csv += metrics.width.toFixed(6) + ",";
            csv += metrics.height.toFixed(6) + ",";
            csv += metrics.netArea.toFixed(8) + ",";
            csv += metrics.grossArea.toFixed(8) + ",";
            csv += metrics.occupancyRatio.toFixed(4) + ",";
            csv += metrics.grossOccupancyRatio.toFixed(4) + ",";
            csv += metrics.holeCount + ",";
            csv += metrics.perimeter.toFixed(6) + ",";
            csv += metrics.anchorCount + ",";
            csv += metrics.anchorDensity.toFixed(2) + ",";
            csv += metrics.perimeterAreaRatio.toFixed(2) + ",";
            csv += metrics.aspectRatio.toFixed(4) + ",";
            csv += metrics.hasFill + ",";
            csv += metrics.hasStroke + ",";
            csv += '"' + metrics.fillColor + '",';
            csv += metrics.strokeWidth.toFixed(6) + "\n";
        }
    }

    // Save CSV to desktop
    var desktop = Folder.desktop;
    var fileName = doc.name.replace(/\.[^\.]+$/, '') + "_analysis.csv";
    var file = new File(desktop + "/" + fileName);

    file.open("w");
    file.write(csv);
    file.close();

    // Show summary
    var summary = "=== DOCUMENT ANALYSIS COMPLETE ===\n\n";
    summary += "Total objects analyzed: " + stats.totalObjects + "\n\n";
    summary += "Objects per layer:\n";

    for (var layerName in stats.byLayer) {
        summary += "  " + layerName + ": " + stats.byLayer[layerName] + "\n";
    }

    summary += "\nCSV saved to:\n" + file.fsName;
    summary += "\n\nOpen this file in Excel or Google Sheets to analyze metrics.";

    alert(summary);
}

// ================================
// ENTRY POINT
// ================================

try {
    if (app.documents.length === 0) {
        alert("Please open your classified document first.");
    } else {
        analyzeDocument();
    }
} catch (err) {
    alert("Error: " + err.message + "\nLine: " + err.line);
}
