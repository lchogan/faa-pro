/**
 * AddTargetLayersBatch.jsx
 *
 * Walks a folder of <code>-diagram.ai files (recursively) and ensures all
 * seven target layers exist in each one:
 *
 *   Taxiways, Footprints, Runways, Lights, Taxiway Labels, Runway Labels, Other
 *
 * Existing layers are left untouched. Missing ones are added empty. Each file
 * is saved in place after the layers are added. Run once before you start
 * manually correcting layer assignments — afterwards, just open each file in
 * Illustrator, drag misclassified objects to the correct layer (use the
 * Layers panel), and save.
 *
 * Usage:
 *   File > Scripts > Other Script > select this file
 *   Pick the folder containing the airports (e.g. .../faa-downloader/airports-class)
 *
 * Hard rule for the labeling pass: do not modify geometry, colors, or the
 * artboard. Only change layer membership. Pathfinder, color edits, transforms,
 * and artboard changes will make these files unusable as ML training data.
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
    var rootFolder = Folder.selectDialog("Select folder containing <code>-diagram.ai files");
    if (!rootFolder) return;

    var aiFiles = findDiagramFiles(rootFolder);
    if (aiFiles.length === 0) {
        alert("No *-diagram.ai files found under:\n" + rootFolder.fsName);
        return;
    }

    var prevAlerts = app.userInteractionLevel;
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

    var processed = 0;
    var totalAdded = 0;
    var perFile = [];
    var errors = [];
    var startTime = new Date().getTime();

    for (var i = 0; i < aiFiles.length; i++) {
        var aiFile = aiFiles[i];
        var doc = null;
        try {
            doc = app.open(aiFile);
            var addedHere = ensureLayers(doc, TARGET_LAYERS);
            if (addedHere.length > 0) {
                doc.save();
                totalAdded += addedHere.length;
                perFile.push(aiFile.name + ": +" + addedHere.length + " (" + addedHere.join(", ") + ")");
            }
            doc.close(SaveOptions.DONOTSAVECHANGES);
            doc = null;
            processed++;
        } catch (e) {
            errors.push(aiFile.name + ": " + e.message);
            if (doc) {
                try { doc.close(SaveOptions.DONOTSAVECHANGES); } catch (ignore) {}
            }
        }
    }

    app.userInteractionLevel = prevAlerts;

    var elapsed = Math.round((new Date().getTime() - startTime) / 1000);
    var msg = "Layer scaffolding complete.\n\n" +
              "Files processed:  " + processed + " / " + aiFiles.length + "\n" +
              "Layers added:     " + totalAdded + "\n" +
              "Elapsed:          " + elapsed + "s\n";
    if (perFile.length > 0) {
        msg += "\nFiles changed:\n  " + perFile.slice(0, 15).join("\n  ");
        if (perFile.length > 15) msg += "\n  ... (+" + (perFile.length - 15) + " more)";
    } else {
        msg += "\nNo changes needed — all files already had the seven target layers.";
    }
    if (errors.length > 0) {
        msg += "\n\nErrors (" + errors.length + "):\n  " + errors.slice(0, 10).join("\n  ");
        if (errors.length > 10) msg += "\n  ... (+" + (errors.length - 10) + " more)";
    }
    alert(msg);
}

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

function ensureLayers(doc, names) {
    var existing = {};
    for (var i = 0; i < doc.layers.length; i++) existing[doc.layers[i].name] = true;
    var added = [];
    for (var j = 0; j < names.length; j++) {
        var name = names[j];
        if (!existing[name]) {
            var layer = doc.layers.add();
            layer.name = name;
            added.push(name);
        }
    }
    return added;
}

main();
