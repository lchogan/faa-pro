#target illustrator

/**
 * PrepareForInspection.jsx
 *
 * Runs at the end of classify.sh on the freshly rendered <airport>-diagram.svg.
 * Saves the SVG as <airport>-diagram.ai, promotes the SVG-import wrapper's
 * children up to the document root (handling both native-sublayer and
 * named-GroupItem variants — AI's SVG importer is inconsistent across
 * versions for Inkscape `groupmode="layer"` groups), ensures a Lights
 * placeholder exists, sorts layers into the standard inspection stack,
 * and locks + hides the reference-only layers. The .ai is saved and left
 * open in Illustrator for the user to review.
 *
 * Driven by /tmp/faa_pro_inspection.json which classify.sh writes:
 *   { "svg_path": "/abs/path/to/<airport>-diagram.svg" }
 *
 * Layer stack top-to-bottom in the Layers panel after this runs:
 *   Runway Labels  → Taxiway Labels → Stars → Lights → Footprints
 *   → Runways → Taxiways → Other → PDF Text Tokens → Metadata
 *
 * Other / PDF Text Tokens / Metadata are locked and hidden.
 */

(function () {

    // ── Configuration ────────────────────────────────────────────────────────

    var CONFIG_PATH = "/tmp/faa_pro_inspection.json";

    // Top-of-panel first. doc.layers[0] is the topmost layer in Illustrator.
    var LAYER_ORDER = [
        "Runway Labels",
        "Taxiway Labels",
        "Stars",
        "Lights",
        "Footprints",
        "Runways",
        "Taxiways",
        "Other",
        "PDF Text Tokens",
        "Metadata"
    ];

    var LOCK_AND_HIDE = ["Other", "PDF Text Tokens", "Metadata"];

    // Temporary name used while promoting children out of the SVG-import
    // wrapper. Picked to never collide with a real target layer.
    var WRAPPER_TEMP_NAME = "__svg_wrapper__";

    // ── Helpers ─────────────────────────────────────────────────────────────
    //
    // Declared before the main flow so we don't rely on ExtendScript's
    // (sometimes inconsistent) function-declaration hoisting.

    function isTargetLayerName(name) {
        for (var i = 0; i < LAYER_ORDER.length; i++) {
            if (LAYER_ORDER[i] === name) return true;
        }
        return false;
    }

    function ensureLayer(doc, name) {
        for (var i = 0; i < doc.layers.length; i++) {
            if (doc.layers[i].name === name) return doc.layers[i];
        }
        var l = doc.layers.add();
        l.name = name;
        return l;
    }

    /** Move every direct pageItem from source into target. */
    function moveAllItems(source, target) {
        var items = source.pageItems;
        for (var i = items.length - 1; i >= 0; i--) {
            items[i].move(target, ElementPlacement.PLACEATEND);
        }
    }

    /**
     * Move a GroupItem's direct children into target, then delete the
     * empty GroupItem. Used when the SVG importer landed our layer-groups
     * as named GroupItems instead of native sublayers.
     */
    function unpackGroupIntoLayer(group, target) {
        var items = group.pageItems;
        for (var i = items.length - 1; i >= 0; i--) {
            items[i].move(target, ElementPlacement.PLACEATEND);
        }
        group.remove();
    }

    /** Recursively flatten a sublayer's nested sublayers and items into target. */
    function collapseSublayerInto(sub, target) {
        while (sub.layers.length > 0) {
            var nested = sub.layers[sub.layers.length - 1];
            nested.locked = false;
            nested.visible = true;
            moveAllItems(nested, target);
            collapseSublayerInto(nested, target);
            nested.remove();
        }
        moveAllItems(sub, target);
    }

    /**
     * A wrapper is a root layer that contains either nested sublayers OR
     * named GroupItems matching one of our target layer names. AI's SVG
     * importer typically produces one such layer (often named "Layer 1"
     * but we've also seen it inherit a sublayer name like "Other").
     */
    function findWrapperLayer(doc) {
        for (var i = 0; i < doc.layers.length; i++) {
            var l = doc.layers[i];
            if (l.layers && l.layers.length > 0) return l;
            if (l.pageItems && l.pageItems.length > 0) {
                for (var j = 0; j < l.pageItems.length; j++) {
                    var item = l.pageItems[j];
                    if (item.typename === 'GroupItem'
                        && item.name
                        && isTargetLayerName(item.name)) {
                        return l;
                    }
                }
            }
        }
        return null;
    }

    /**
     * Promote everything inside a wrapper layer to the document root.
     * Handles both native sublayers and named GroupItems — AI's SVG
     * importer can produce either depending on version + content.
     */
    function promoteFromWrapper(wrapper, doc) {

        // ── Native sublayer path ────────────────────────────────────────
        //
        // Primary: Layer.move(doc, PLACEATBEGINNING) atomically promotes
        // the sublayer (and all its content) to the document root.
        //
        // Fallback: some AI versions throw on Layer.move across the
        // root boundary. In that case we create a fresh root layer with
        // the same name, copy the sub's content into it, and delete the
        // sub. Either way the artwork ends up at root.

        while (wrapper.layers.length > 0) {
            var sub = wrapper.layers[wrapper.layers.length - 1];
            var origName = sub.name;
            sub.locked = false;
            sub.visible = true;

            var moved = false;
            try {
                sub.move(doc, ElementPlacement.PLACEATBEGINNING);
                moved = true;
            } catch (e) {
                moved = false;
            }

            if (!moved) {
                var rootLayer = ensureLayer(doc, origName);
                collapseSublayerInto(sub, rootLayer);
                try { sub.remove(); } catch (e) {}
            }
        }

        // ── Named-GroupItem path ────────────────────────────────────────
        //
        // Snapshot the matching groups first — wrapper.pageItems is live,
        // and unpacking modifies it.

        var namedGroups = [];
        for (var i = 0; i < wrapper.pageItems.length; i++) {
            var item = wrapper.pageItems[i];
            if (item.typename === 'GroupItem'
                && item.name
                && isTargetLayerName(item.name)) {
                namedGroups.push(item);
            }
        }
        for (var i = 0; i < namedGroups.length; i++) {
            var g = namedGroups[i];
            var rootLayer = ensureLayer(doc, g.name);
            unpackGroupIntoLayer(g, rootLayer);
        }

        // ── Orphan items ────────────────────────────────────────────────
        //
        // Anything still inside the wrapper (unnamed groups, stray paths)
        // gets dumped to Other rather than silently dropped.

        if (wrapper.pageItems.length > 0) {
            var otherLayer = ensureLayer(doc, "Other");
            moveAllItems(wrapper, otherLayer);
        }
    }

    function promoteFromWrappers(doc) {
        var safety = 20;
        while (safety-- > 0) {
            var wrapper = findWrapperLayer(doc);
            if (!wrapper) return;

            wrapper.name = WRAPPER_TEMP_NAME;
            wrapper.locked = false;
            wrapper.visible = true;

            promoteFromWrapper(wrapper, doc);

            if (wrapper.pageItems.length === 0 && wrapper.layers.length === 0) {
                wrapper.remove();
            } else {
                // Wrapper still holds content we couldn't relocate — break
                // out rather than spin forever.
                break;
            }
        }
    }

    // ── Main flow ───────────────────────────────────────────────────────────

    var configFile = new File(CONFIG_PATH);
    if (!configFile.exists) {
        alert("PrepareForInspection: config not found at " + CONFIG_PATH);
        return;
    }
    configFile.open("r");
    var jsonStr = configFile.read();
    configFile.close();

    var config;
    try {
        config = eval('(' + jsonStr + ')');
    } catch (e) {
        alert("PrepareForInspection: bad JSON in " + CONFIG_PATH + " — " + e.message);
        return;
    }

    if (!config || !config.svg_path) {
        alert("PrepareForInspection: config missing svg_path");
        return;
    }

    var svgFile = new File(config.svg_path);
    if (!svgFile.exists) {
        alert("PrepareForInspection: SVG not found at " + config.svg_path);
        return;
    }

    var prevAlerts = app.userInteractionLevel;
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

    var doc;
    try {
        doc = app.open(svgFile);
    } catch (e) {
        app.userInteractionLevel = prevAlerts;
        alert("PrepareForInspection: failed to open SVG — " + e.message);
        return;
    }

    try {

        // 1. Save As .ai so the original SVG stays untouched.
        var aiPath = svgFile.fsName.replace(/\.[^.]+$/, '.ai');
        var aiFile = new File(aiPath);
        var aiOpts = new IllustratorSaveOptions();
        aiOpts.compatibility = Compatibility.ILLUSTRATOR17;
        aiOpts.pdfCompatible = true;
        doc.saveAs(aiFile, aiOpts);

        // 2. Promote SVG-import wrapper's children to the root.
        promoteFromWrappers(doc);

        // 3. Ensure every target layer exists (creates empty placeholders
        //    for any class with no polygons + the manual-only Lights layer).
        for (var i = 0; i < LAYER_ORDER.length; i++) {
            ensureLayer(doc, LAYER_ORDER[i]);
        }

        // 4. Reorder. doc.layers is a stack with [0] = topmost. Bring the
        //    first entry to front, chain the rest after it via PLACEAFTER.
        var prev = null;
        for (var i = 0; i < LAYER_ORDER.length; i++) {
            var layer = doc.layers.getByName(LAYER_ORDER[i]);
            layer.locked = false;
            layer.visible = true;
            if (i === 0) {
                layer.zOrder(ZOrderMethod.BRINGTOFRONT);
            } else {
                layer.move(prev, ElementPlacement.PLACEAFTER);
            }
            prev = layer;
        }

        // 5. Lock and hide reference-only layers.
        for (var i = 0; i < LOCK_AND_HIDE.length; i++) {
            var l;
            try { l = doc.layers.getByName(LOCK_AND_HIDE[i]); }
            catch (e) { continue; }
            l.visible = false;
            l.locked  = true;
        }

        // 6. Save and leave open.
        doc.save();

        app.userInteractionLevel = prevAlerts;

    } catch (err) {
        app.userInteractionLevel = prevAlerts;
        alert("PrepareForInspection: " + err.message + (err.line ? " (line " + err.line + ")" : ""));
    }

}());
