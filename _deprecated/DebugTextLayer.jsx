/**
 * DebugTextLayer.jsx
 *
 * Reads a *_text_debug.json file produced by dump_pdf_text.py and places
 * every extracted PDF word as a small red text frame on a "PDF Text Debug"
 * layer in the currently open document.
 *
 * Use this to verify that the PDF text coordinates line up with the vector
 * paths — every visible letter/number on the diagram should have a red label
 * sitting on top of it.
 *
 * Usage:
 *   File > Scripts > Other Script > DebugTextLayer.jsx
 *   Pick the *_text_debug.json file when prompted.
 *   To remove the layer: delete "PDF Text Debug" in the Layers panel.
 */

var LAYER_NAME = "PDF Text Debug";
var FONT_SIZE  = 5;   // pt — small enough not to obscure the paths underneath

function main() {
    if (app.documents.length === 0) {
        alert("Open the airport diagram in Illustrator first.");
        return;
    }
    var doc = app.activeDocument;

    var jsonFile = File.openDialog("Select *_text_debug.json", "*.json");
    if (!jsonFile) return;

    var payload = readJSON(jsonFile);
    if (!payload || !payload.words) {
        alert("Could not read JSON or no words found.");
        return;
    }

    var prevAlerts = app.userInteractionLevel;
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

    // Remove any existing debug layer so re-runs are clean.
    for (var li = doc.layers.length - 1; li >= 0; li--) {
        if (doc.layers[li].name === LAYER_NAME) {
            doc.layers[li].locked = false;
            doc.layers[li].remove();
            break;
        }
    }

    var layer = doc.layers.add();
    layer.name = LAYER_NAME;
    // Move to top so labels sit above everything.
    layer.zOrder(ZOrderMethod.BRINGTOFRONT);

    // The JSON coordinates are in PDF-page space flipped to Illustrator's
    // y-up frame (origin at bottom-left of the page).  The artboard in AI
    // also has its origin at the page bottom-left, so coordinates should
    // match directly.  If you see a constant offset, adjust OFFSET_X/Y.
    var OFFSET_X = 0;
    var OFFSET_Y = 0;

    var ab = doc.artboards[0].artboardRect; // [left, top, right, bottom]
    // If the artboard doesn't start at (0,0) we need to shift.
    OFFSET_X = ab[0];           // artboard left  (usually 0)
    OFFSET_Y = ab[3];           // artboard bottom (usually 0, can be negative)

    var words = payload.words;
    var placed = 0;

    for (var i = 0; i < words.length; i++) {
        var w = words[i];
        // Place text at the top-left of the word bounding box in AI coords.
        var x = w.x_min + OFFSET_X;
        var y = w.y_min + OFFSET_Y;  // y_min ≈ baseline in AI space

        try {
            var tf = layer.textFrames.pointText([x, y]);
            tf.contents = w.text;
            tf.textRange.characterAttributes.size = FONT_SIZE;

            // Red fill so it stands out against the diagram.
            var col = new RGBColor();
            col.red = 220; col.green = 0; col.blue = 0;
            tf.textRange.characterAttributes.fillColor = col;
            placed++;
        } catch (e) {
            // Skip any item that can't be placed.
        }
    }

    layer.locked = true;  // prevent accidental edits
    app.userInteractionLevel = prevAlerts;

    alert("Placed " + placed + " of " + words.length + " text items on \"" +
          LAYER_NAME + "\".\n\nZoom into any label area and check that the " +
          "red text sits on top of the corresponding vector path.");
}

function readJSON(file) {
    file.encoding = "UTF-8";
    if (!file.open("r")) {
        alert("Could not open: " + file.fsName);
        return null;
    }
    var text = file.read();
    file.close();
    try {
        if (typeof JSON !== "undefined" && JSON.parse) return JSON.parse(text);
        return (new Function("return (" + text + ")"))();
    } catch (e) {
        alert("JSON parse error: " + e.message);
        return null;
    }
}

main();
