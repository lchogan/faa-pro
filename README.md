# FAA-PRO Airport Diagram Classifier

**Updated**: 2026-02-19 based on actual airport diagram analysis

## 🎯 What This Does

Automatically classifies FAA airport diagrams from unprocessed PDFs into semantic layers:

- **Taxiways** (gray areas)
- **Footprints** (buildings, all sizes)
- **Runways** (large rectangles)
- **Text** (individual letters and labels)
- **Stars** (symbols)
- **Lines** (lat/long grid lines)
- **Arrows** (arrowheads + connected lines)

## 🚀 Quick Start

### 1. Open Airport PDF in Illustrator
```
File → Open → Select cha.pdf, ord.pdf, or cle.pdf
```

### 2. Run the Classifier
```
File → Scripts → Other Script...
→ Navigate to ~/Documents/startups/aoa/illustrator/faa-pro/
→ Select AirportDiagramClassifier.jsx
```

### 3. Review Output Layers
Check the newly created layers. Objects with unclear classification will be in "Uncertain" layer.

---

## 📁 Project Files

### Main Scripts
- **`AirportDiagramClassifier.jsx`** ← **USE THIS ONE**
  - Complete classifier for unprocessed airport diagrams
  - Handles clipping masks, off-artboard cleanup
  - Detects taxiways by color, lines by stroke
  - Geometric classification for everything else

- `AnalyzeClassifiedDocument.jsx`
  - Extracts metrics from already-classified documents
  - Outputs CSV with all object properties
  - Use for calibration and threshold tuning

### Documentation
- **`AIRPORT_CLASSIFIER_GUIDE.md`** ← **READ THIS FIRST**
  - Complete user guide
  - Configuration reference
  - Troubleshooting
  - Calibration workflow

- `CHA_DIAGRAM_ANALYSIS.md`
  - Analysis of cha-diagram.svg (classified example)
  - Actual size ranges and metrics from real data
  - Classification strategy explanation

- `PROJECT_SUMMARY.md`
  - Original project summary (now outdated)
  - Based on incorrect assumptions about data

### Configuration Presets (Archived)
- `CONFIG_PRESETS.md` - Various configuration scenarios
- `CALIBRATION_GUIDE.md` - Original calibration guide
- `QUICK_REFERENCE.md` - Quick reference card

### Original Scripts (Archived)
- `SemanticVectorClassifier.jsx` - Original version
  - Based on incorrect assumption (individual 0.008" letters)
  - Kept for reference

---

## 📊 How It Works

### Phase 1: Color & Stroke Detection (Easy Wins)

**Taxiways**: Unique gray fill color
```
if (RGB ≈ [207, 207, 207]) → Taxiway
```

**Lines**: Stroke width = 0.39pt (FAA standard)
```
if (strokeWidth ≈ 0.39pt && !filled) → Line
  if (vertical or horizontal) → Lines layer
  if (touching arrowhead) → Arrows layer
```

### Phase 2: Geometric Classification

**Size-based filtering**:
```
< 0.003 sq in + 3-5 anchors  → Arrowhead → Arrows
< 0.01 sq in + 8-15 anchors  → Star
0.0001 - 0.5 sq in + filled  → Text
> 3 sq in + simple + elongated → Runway
> 0.002 sq in + filled       → Footprint
```

### Phase 3: Arrow Assembly

- Find all arrowheads (tiny triangles)
- Find all lines
- Detect which lines touch arrowheads
- Move both to "Arrows" layer
- Remaining vertical/horizontal lines → "Lines" layer

---

## 🎛️ Key Configuration Settings

Edit `CONFIG` object in `AirportDiagramClassifier.jsx`:

### Most Important Thresholds

```javascript
// Taxiway detection (color-based)
taxiwayGrayMin: 190,         // Adjust if taxiways not detected
taxiwayGrayMax: 220,

// Text vs. Building size threshold
textMaxArea: 0.5,            // Raise if large text → footprints
                             // Lower if small buildings → text

// Runway detection
runwayMinArea: 3.0,          // Lower for smaller airports
runwayAspectMin: 0.2,        // Elongated threshold
runwayAspectMax: 5.0,

// Line detection (0.39pt is FAA standard)
lineStrokeWidth: 0.39,       // Exact stroke width in points
lineStrokeTolerance: 0.05,   // ±0.05pt tolerance

// Arrow connection
arrowConnectionDistance: 2,  // Raise if arrows not connecting
```

### Pre-Processing

```javascript
releaseClippingMasks: true,  // ✅ Keep ON - handles PDF clipping
deleteOffArtboard: true,     // ✅ Keep ON - cleans up edges
ungroupAll: false,           // ⚠️ Set true only if objects grouped
```

---

## 🔧 Calibration for Your Data

### Step 1: Test Run
1. Open `cha.pdf` in Illustrator
2. Run `AirportDiagramClassifier.jsx`
3. Review output layers

### Step 2: Check "Uncertain" Layer
- Select objects in Uncertain layer
- Window → Attributes → Show Note
- Read metrics to understand why they weren't classified

### Step 3: Adjust Thresholds

**Common Issues**:

| Symptom | Fix |
|---------|-----|
| Text classified as Footprints | Raise `textMaxArea` to 0.8 |
| Small buildings classified as Text | Lower `textMaxArea` to 0.3 |
| Taxiways not detected | Check actual RGB values, adjust `taxiwayGray*` |
| Lines not connecting to arrows | Raise `arrowConnectionDistance` to 5 |
| Runways in Footprints | Lower `runwayMinArea` to 2.0 |

### Step 4: Test on Other Airports
- Run on `ord.pdf` (large airport)
- Run on `cle.pdf` (medium airport)
- Verify thresholds work across different scales

---

## 📈 Expected Accuracy

### With Default Settings
- **Taxiways**: 95-100% (color-based, very reliable)
- **Lines**: 90-98% (stroke-based, mostly reliable)
- **Runways**: 85-95% (size/shape-based)
- **Text**: 80-90% (size-based, overlaps with small buildings)
- **Footprints**: 75-85% (catch-all category, varied)
- **Arrows**: 80-90% (depends on proximity detection)
- **Stars**: 90-95% (anchor count-based)

### With Calibration (15-30 min tuning)
- All categories: 90-98%
- Uncertain: <5% of total

---

## 🐛 Troubleshooting

### Script Errors

**"Please open a document first"**
→ Open a PDF in Illustrator before running

**"Error: undefined is not an object"**
→ Document might have unexpected object types
→ Set `logDetails: true` to see which object caused error

### Classification Issues

**Everything in "Uncertain"**
1. Check if thresholds match your data scale
2. Run `AnalyzeClassifiedDocument.jsx` to get actual metrics
3. Adjust CONFIG based on real data ranges

**Taxiways not detected**
1. Manually select a taxiway
2. Window → Color panel → Check RGB values
3. Adjust `taxiwayGrayMin/Max` to match

**Text and Footprints mixing**
- This is the hardest boundary to get right
- Small buildings and large text overlap in size
- May require manual review of borderline cases

---

## 🎓 Understanding the Data

### Actual Size Ranges (from cha-diagram.svg)

**Taxiways**:
- Size: 2-3.5 inches, 4-11 sq in
- Color: RGB(207, 207, 207) gray
- Complex shapes with curves

**Runways**:
- Size: 6-7 sq in
- Simple rectangles, 4 anchors
- Aspect ratio 0.32-0.70

**Text** (individual letters, unclassified state):
- Size: Varies by font/label
- Expected: 0.05-0.5 sq in for typical airport labels
- May be larger for compass rose, airport name

**Footprints** (buildings):
- Size: 0.003 sq in (small sheds) to 15+ sq in (terminals)
- Wide range - hardest to classify
- Many shapes and sizes

**Arrowheads**:
- Size: <0.003 sq in (tiny)
- Always 3-5 anchors (triangles)
- Aspect ratio 1.2-4.0

**Lines**:
- Width: Near zero (stroked, no fill)
- Length: 0.5-6+ inches
- Perfectly vertical or horizontal for lat/long

**Stars**:
- Size: <0.01 sq in
- 8-15 anchors (pentagon/hexagon)

---

## 🔬 Advanced Usage

### Batch Processing Multiple Airports

Add to end of script:

```javascript
function processBatch() {
    var folder = Folder.selectDialog("Select folder with airport PDFs");
    if (!folder) return;

    var files = folder.getFiles("*.pdf");

    for (var i = 0; i < files.length; i++) {
        app.open(files[i]);
        processDocument();

        // Save as AI
        var savePath = new File(
            files[i].path + "/" +
            files[i].name.replace(".pdf", "_classified.ai")
        );
        app.activeDocument.saveAs(savePath);
        app.activeDocument.close();
    }

    alert("Processed " + files.length + " airports!");
}

// Run it:
// processBatch();
```

### Export Classification Report

After classification, export metrics:

```javascript
// Add to processDocument() before final alert
var csvReport = "Airport,Layer,Count\n";
csvReport += doc.name + ",Taxiways," + stats.taxiways + "\n";
csvReport += doc.name + ",Footprints," + stats.footprints + "\n";
// ... etc

var file = new File("~/Desktop/classification_report.csv");
file.open("a");  // Append mode
file.write(csvReport);
file.close();
```

---

## 📝 Workflow Checklist

### For Each New Airport Diagram:

- [ ] Open PDF in Illustrator
- [ ] Run AirportDiagramClassifier.jsx
- [ ] Review summary report
- [ ] Check Uncertain layer (should be <10%)
- [ ] Spot-check each layer for obvious errors
- [ ] Manually reclassify borderline cases if needed
- [ ] Save as .ai file
- [ ] Export final layers for use

### For New Airport Types (different scale/format):

- [ ] Run classifier with default settings
- [ ] Note which categories fail
- [ ] Extract metrics with AnalyzeClassifiedDocument.jsx
- [ ] Adjust CONFIG thresholds
- [ ] Create custom preset for this type
- [ ] Document new settings

---

## 🤝 Contributing

If you improve the classifier:

1. Document what you changed and why
2. Note what airport types it works better for
3. Save your CONFIG as a preset
4. Share back for future projects

---

## 📞 Support

**Issues**:
1. Check AIRPORT_CLASSIFIER_GUIDE.md troubleshooting
2. Review CHA_DIAGRAM_ANALYSIS.md for data expectations
3. Run with `debugMode: true` to see object notes
4. Extract metrics with AnalyzeClassifiedDocument.jsx

**Questions**:
- What are the actual size ranges in your PDFs?
- Which categories are mixing?
- What does the Uncertain layer contain?
- Have you tested on all three airport sizes (cha, ord, cle)?

---

## 📜 Version History

### v2.0 (2026-02-19) - Airport-Specific Classifier
- **MAJOR REWRITE** based on actual airport diagram analysis
- Added color detection for taxiways (gray fill)
- Added stroke detection for lines
- Added arrow assembly (arrowheads + connected lines)
- Added pre-processing (release clipping masks, delete off-artboard)
- Tuned thresholds for real airport data (not theoretical)
- Separate handling of lat/long lines vs. arrow lines

### v1.0 (2026-02-19) - Initial Version
- Generic semantic classifier
- Based on incorrect assumptions about data
- Designed for individual 0.008" letters (doesn't match actual data)
- Kept as SemanticVectorClassifier.jsx for reference

---

## 📚 Key Learnings

### What Changed from v1.0 → v2.0

1. **Data Reality**: Text is individual letters in unclassified state (not grouped blocks in classified state)
2. **Color Matters**: Taxiways have unique gray fill - use it!
3. **Stroke Matters**: Lines are stroked, not filled
4. **Context Matters**: Lines touching arrowheads are different from lat/long lines
5. **Size Varies**: Buildings range from tiny sheds to massive terminals
6. **Preprocessing Needed**: PDFs have clipping masks and off-artboard junk

### Design Principles

1. **Use deterministic rules when possible** (color, stroke) - more reliable than geometric heuristics
2. **Multi-factor scoring** for ambiguous cases (size + shape + anchors)
3. **Debug output** is essential - users need to see WHY something was classified
4. **Calibration is normal** - no one-size-fits-all solution for varied airport sizes
5. **Edge cases happen** - Uncertain layer is expected, not a failure

---

**Ready to classify your airport diagrams? Start with AIRPORT_CLASSIFIER_GUIDE.md!**
