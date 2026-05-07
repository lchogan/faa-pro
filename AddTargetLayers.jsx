/**
 * AddTargetLayers.jsx
 *
 * Lightweight helper for the manual labeling workflow. Run this on a document
 * that has already been classified (e.g. by AirportDiagramClassifier.jsx) to
 * ensure all seven target layers exist:
 *
 *   Taxiways, Footprints, Runways, Lights, Taxiway Labels, Runway Labels, Other
 *
 * Layers that already exist are kept untouched. Missing ones are created
 * empty. After this runs you can use the Layers panel to drag objects between
 * layers as needed, then save.
 *
 * Hard rule for the labeling pass: do not modify geometry, colors, or the
 * artboard. Only change layer membership. Anything that gets pathfinder-united,
 * recolored, transformed, or moved becomes useless as ML training data.
 */

var TARGET_LAYERS = [
    "Taxiways",
    "Footprints",
    "Runways",
    "Lights",
    "Taxiway Labels",
    "Runway Labels",
    "Stars",
    "Other"
];

function main() {
    if (app.documents.length === 0) {
        alert("Open a document first.");
        return;
    }
    var doc = app.activeDocument;
    var existing = {};
    for (var i = 0; i < doc.layers.length; i++) existing[doc.layers[i].name] = true;

    var added = [];
    for (var j = 0; j < TARGET_LAYERS.length; j++) {
        var name = TARGET_LAYERS[j];
        if (!existing[name]) {
            var layer = doc.layers.add();
            layer.name = name;
            added.push(name);
        }
    }

    if (added.length === 0) {
        alert("All seven target layers already exist.");
    } else {
        alert("Added missing layers:\n  " + added.join("\n  "));
    }
}

main();
