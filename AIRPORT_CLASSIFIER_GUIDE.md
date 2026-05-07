# Airport Diagram Classifier - User Guide

## Overview

This script processes **unclassified** FAA airport diagrams and automatically classifies objects into semantic layers.

**Input**: Everything in one layer, with clipping masks and off-artboard elements
**Output**: 7 organized layers with classified elements

---

## Quick Start

### 1. Prepare Document
- Open your airport diagram PDF in Illustrator (File → Open)
- The document should have everything in one layer (typical for raw PDFs)

### 2. Run Script
- File → Scripts → Other Script...
- Navigate to: `~/Documents/startups/aoa/illustrator/faa-pro/`
- Select `AirportDiagramClassifier.jsx`
- Click "Open"

### 3. Review Results
The script will create these layers:
- **Taxiways** - Gray-filled areas
- **Footprints** - Buildings (all sizes)
- **Runways** - Large rectangular areas
- **Text** - Individual letters and labels
- **Stars** - Star symbols
- **Lines** - Vertical/horizontal lat/long lines
- **Arrows** - Arrowheads + their attached lines
- **Uncertain** - Objects that don't fit clear categories

---

## What the Script Does

### Phase 1: Pre-Processing

1. **Release Clipping Masks** ✅
   - Automatically unlocks all masked groups
   - No manual unmasking needed

2. **Delete Off-Artboard Objects** ✅
   - Removes elements outside the page boundaries
   - Cleans up stray objects

3. **Ungroup (Optional)** ⚠️
   - Set `ungroupAll: true` in CONFIG if needed
   - Default: OFF (usually not needed for PDFs)

### Phase 2: Classification

Uses a **multi-factor decision tree**:

```
┌─ Gray Fill? ────────────→ TAXIWAY
│
├─ Stroked + No Fill? ────→ LINE or ARROW (if touching arrowhead)
│
├─ Tiny (<0.003 sq in) + 3-5 anchors? ────→ ARROWHEAD → ARROWS layer
│
├─ Small (<0.01 sq in) + 8-15 anchors? ───→ STAR
│
├─ Large (>3 sq in) + Simple + Elongated? →RUNWAY
│
├─ Small-Medium (0.0001-0.5 sq in)? ──────→ TEXT
│
├─ Medium-Large (>0.002 sq in) + Filled? ─→ FOOTPRINT
│
└─ Else ──────────────────────────────────→ UNCERTAIN
```

### Phase 3: Arrow Detection

**Special Logic**:
- Finds all arrowheads (tiny triangles)
- Finds all lines
- Checks which lines touch arrowheads
- Moves both to "Arrows" layer
- Remaining lines (lat/long) stay in "Lines" layer

---

## Configuration

Edit the `CONFIG` object at the top of the script:

### Pre-Processing
```javascript
releaseClippingMasks: true,  // Auto-release masks
deleteOffArtboard: true,     // Clean up off-page objects
ungroupAll: false,           // Set true if objects are grouped
```

### Color Detection (Taxiways)
```javascript
taxiwayGrayMin: 190,         // RGB min for gray (0-255)
taxiwayGrayMax: 220,         // RGB max for gray
taxiwayGrayTolerance: 15,    // R≈G≈B tolerance
```

**How it works**: Taxiways use gray fill (#CFCFCF ≈ RGB(207,207,207))
- Checks if R, G, B are all in range [190, 220]
- Checks if R≈G≈B (within 15 units) to ensure grayscale

### Size Thresholds
```javascript
arrowheadMaxArea: 0.003,     // sq in - tiny triangles
starMaxArea: 0.01,           // sq in - small symbols
textMaxArea: 0.5,            // sq in - individual letters
textMinArea: 0.00005,        // sq in - filter artifacts
footprintMinArea: 0.002,     // sq in - small sheds
runwayMinArea: 3.0,          // sq in - large rectangles
```

**Tuning tips**:
- If small buildings classified as text: **Lower** `textMaxArea`
- If large text classified as footprints: **Raise** `textMaxArea`
- If tiny artifacts appear: **Raise** `textMinArea`

### Aspect Ratio
```javascript
runwayAspectMin: 0.2,        // Aspect < 0.2 = very tall
runwayAspectMax: 5.0,        // Aspect > 5.0 = very wide
arrowheadAspectMin: 1.2,     // Triangular shapes
arrowheadAspectMax: 4.0,
```

**Aspect = Width / Height**:
- Aspect < 1.0 = Taller than wide (portrait)
- Aspect > 1.0 = Wider than tall (landscape)

### Anchor Count
```javascript
arrowheadAnchorCount: [3, 4, 5],  // Triangles
starAnchorMin: 8,                 // Pentagon/hexagon stars
starAnchorMax: 15,
runwayAnchorMax: 8,               // Simple rectangles
```

### Line Detection
```javascript
lineStrokeWidth: 0.39,       // Exact stroke width for lines (points)
lineStrokeTolerance: 0.05,   // Tolerance for stroke width matching
lineMaxArea: 0.02,           // Lines have minimal area
lineMinLength: 0.5,          // Minimum 0.5 inches
lineAngleTolerance: 2,       // ±2° from vertical/horizontal
```

**Line Detection Logic**:
- Lines must have stroke width of **exactly 0.39 points** (±0.05 tolerance)
- This is the standard line weight in FAA airport diagrams
- No fill or minimal fill area

**Lat/Long Lines**:
- Must be within 2° of 0°, 90°, 180°, or 270°
- Perfectly vertical or horizontal grid lines

### Arrow Connection
```javascript
arrowConnectionDistance: 2,  // Points - proximity for "touching"
```

If arrowheads aren't connecting to lines, **increase** this value.

### Debug Options
```javascript
debugMode: true,             // Write metrics to object notes
showReport: true,            // Show summary dialog
logDetails: false,           // Verbose logging to console
```

---

## Calibration Workflow

### Step 1: Run with Defaults
1. Process one airport (e.g., cha.pdf)
2. Review the output layers
3. Note which objects are misclassified

### Step 2: Identify Patterns

**Common Issues**:

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Text in Footprints | Text too large | Raise `textMaxArea` to 0.8 |
| Footprints in Text | Buildings too small | Lower `textMaxArea` to 0.3 |
| Runways in Footprints | Runway threshold too high | Lower `runwayMinArea` to 2.0 |
| Arrowheads in Uncertain | Wrong size/shape | Adjust `arrowheadMaxArea` |
| Lines not vertical/horizontal | Angle tolerance too tight | Raise `lineAngleTolerance` to 5 |
| Arrows not connecting | Distance too small | Raise `arrowConnectionDistance` to 5 |

### Step 3: Check "Uncertain" Layer

Objects in "Uncertain" didn't match any rule. Check their notes (Window → Attributes → Show Note) to see why.

### Step 4: Test on Other Airports

- Run on ord.pdf (large airport - O'Hare)
- Run on cle.pdf (medium airport - Cleveland)
- Verify thresholds work across different sizes

---

## Airport-Specific Variations

### Small Regional Airports
- Fewer taxiways
- Smaller runways → May need to lower `runwayMinArea`
- Less text

### Large International Airports (ORD, LAX)
- Complex taxiway networks
- Multiple runways
- Dense text labels → May need wider `textMaxArea`

### Military Airports
- May have unique symbols → Check Stars/Uncertain layers
- Different taxiway colors? → Adjust `taxiwayGray` thresholds

---

## Troubleshooting

### "Everything goes to Uncertain"
**Cause**: Thresholds don't match your data
**Fix**:
1. Set `debugMode: true`
2. Select 5-10 objects from Uncertain
3. Read their notes to see actual metrics
4. Adjust CONFIG thresholds based on real data

### "Taxiways not detected"
**Cause**: Different gray color in your PDFs
**Fix**:
1. Select a taxiway manually
2. Window → Color → Check RGB values
3. Adjust `taxiwayGrayMin/Max` to match

### "Lines classified as Footprints"
**Cause**: Lines have fill instead of just stroke
**Fix**: Lines should be stroked only. If they have fill, they'll be classified by size.

### "Small buildings disappearing"
**Cause**: Being classified as Text
**Fix**: Lower `textMaxArea` or check if `footprintMinArea` is too high

### "Arrowheads and lines not connecting"
**Cause**: They're not truly adjacent
**Fix**:
1. Increase `arrowConnectionDistance` to 5-10 points
2. Or manually move misclassified lines

### "Script very slow"
**Cause**: Large document (1000+ objects)
**Fix**:
1. Set `logDetails: false` (console logging is slow)
2. Set `ungroupAll: false` if not needed
3. Process in batches (select portions of doc)

---

## Advanced: Custom Rules

### Add a New Layer Type

Want to classify "Parking Areas" separately?

1. Add to CONFIG:
```javascript
layerParking: "Parking",
parkingMinArea: 1.0,
parkingMaxArea: 5.0,
```

2. Create layer in `processDocument()`:
```javascript
var layers = {
    // ... existing layers ...
    parking: getOrCreateLayer(doc, CONFIG.layerParking)
};
```

3. Add classification logic in `classifyItem()`:
```javascript
// After runway check, before footprints
if (item.filled && areaIn >= CONFIG.parkingMinArea && areaIn <= CONFIG.parkingMaxArea) {
    // Additional logic: check if near terminal, has striping pattern, etc.
    return {
        type: "parking",
        reason: "Parking area detected",
        metrics: metrics
    };
}
```

4. Add to stats and layer assignment.

### Detect Striped Patterns

Some taxiways/parking have diagonal lines:

```javascript
function hasStripedPattern(item) {
    // Check if compound path with multiple sub-paths
    if (item.typename === "CompoundPathItem" && item.pathItems.length > 5) {
        // Count parallel lines
        // ... implement pattern detection ...
        return true;
    }
    return false;
}
```

---

## Batch Processing

To process multiple airports:

```javascript
// Add to end of script:
function processBatch(folderPath) {
    var folder = new Folder(folderPath);
    var files = folder.getFiles("*.pdf");

    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        app.open(file);
        processDocument();
        app.activeDocument.saveAs(
            new File(file.path + "/" + file.name.replace(".pdf", "_classified.ai"))
        );
        app.activeDocument.close();
    }
}

// Usage:
// processBatch("~/Documents/startups/aoa/code/foo-pro/");
```

---

## Performance Tips

1. **Turn off debug mode** for production runs:
   ```javascript
   debugMode: false,
   showReport: true,  // Keep report
   logDetails: false
   ```

2. **Process in batches**: If doc has 5000+ objects, select 1000 at a time

3. **Pre-clean in Illustrator**:
   - Delete unnecessary layers manually first
   - Simplify complex paths (Object → Path → Simplify)

---

## Output Validation Checklist

After classification, verify:

- [ ] All taxiways are gray-filled
- [ ] Runways are large, simple rectangles
- [ ] Text contains individual letters (not buildings)
- [ ] Lines are vertical/horizontal
- [ ] Arrowheads are in Arrows layer with their lines
- [ ] Stars are symbol shapes
- [ ] Footprints are buildings (all sizes)
- [ ] Uncertain layer < 10% of total objects

---

## Next Steps

1. **Run on test airports** (cha, ord, cle)
2. **Calibrate thresholds** for your specific data
3. **Document your CONFIG** for future use
4. **Create airport-specific presets** if needed
5. **Build batch workflow** for processing multiple diagrams

---

## Files in This Package

- `AirportDiagramClassifier.jsx` - Main classifier script
- `AIRPORT_CLASSIFIER_GUIDE.md` - This guide
- `CHA_DIAGRAM_ANALYSIS.md` - Analysis of classified example
- `SemanticVectorClassifier.jsx` - Original script (archived)
- `AnalyzeClassifiedDocument.jsx` - Metrics extraction tool

---

**Version**: 2.0 (Airport-specific)
**Date**: 2026-02-19
**Project**: AOA FAA-PRO
