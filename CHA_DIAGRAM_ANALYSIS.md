# CHA Diagram Analysis - Actual Data Structure

**Date**: 2026-02-19
**Source**: cha-diagram.svg (classified example)

## Document Info
- **Size**: 5.38" × 8.25" (387.39 × 593.74 points)
- **Total Objects**: 794
- **Layers**: 7 classified categories

---

## Layer Breakdown

### 1. TAXIWAYS (46 objects)
**Fill**: #cfcfcf (light gray) - `cls-2`
**Stroke**: None

**Characteristics**:
- **Size**: 2-3.5 inches (LARGE)
- **Area**: 4-11 sq in
- **Aspect Ratio**: 1.09-1.43 (roughly square/rectangular)
- **Anchors**: 11-96 (complex shapes with curves)
- **Easy Identifier**: **GRAY FILL COLOR** ✅

**Classification Strategy**:
```javascript
if (fillColor == "#cfcfcf" || fillColor == "RGB(207,207,207)") {
    return "Taxiway";
}
```

---

### 2. FOOTPRINTS (28 objects - Buildings)
**Fill**: Black (evenodd fill-rule) - `cls-3`
**Stroke**: None

**Characteristics**:
- **Size**: MIXED - from 0.003 sq in to 15+ sq in
  - Large footprints: 12-15 sq in (actual buildings)
  - Tiny footprints: 0.003-0.05 sq in (possibly symbols/markers)
- **Aspect Ratio**: 0.41-1.23 (varied)
- **Anchors**: 4-40

**Problem**: Same class (cls-3) as Text, Runways, Arrowheads!

---

### 3. RUNWAYS (2 objects)
**Fill**: Black - `cls-3`
**Stroke**: None

**Characteristics**:
- **Size**: 6-7 sq in (large rectangles)
- **Dimensions**: 1.5-2.2 in × 3-4.8 in
- **Aspect Ratio**: 0.32-0.70 (tall/vertical rectangles)
- **Anchors**: 4 (simple rectangles)

**Classification Strategy**:
- Large size (>5 sq in)
- Aspect ratio < 0.8 (taller than wide) OR > 1.25 (wider than tall)
- Very few anchors (4-6)

---

### 4. TEXT (682 objects - 85% of all objects!)
**Fill**: Mostly black (`cls-3`), some white (`cls-1`)
**Stroke**: None

**CRITICAL DISCOVERY**:
❌ **NOT individual letters!**
✅ **Text groups/blocks** - entire words, labels, or grouped text elements

**Characteristics**:
- **Size**: HUGE - 2.7 to 17+ sq in bounding boxes
- **Dimensions**: 0.5-5 inches
- **Aspect Ratio**: 0.99-8.03 (wildly varied)
  - Some nearly square (compass rose text around circle)
  - Some very wide (horizontal labels)
- **Anchors**: 10-76

**Implication**: The original script design based on 0.008" × 0.063" individual letters is **completely wrong** for this data!

---

### 5. STARS (3 objects - Symbols)
**Fill**: White (`cls-1`) or Black (`cls-3`)
**Stroke**: None

**Characteristics**:
- **Size**: 0.0026-2.97 sq in (one large, two tiny)
- Large star: 2.33" × 1.27"
- Tiny stars: ~0.05" × 0.05"
- **Anchors**: 10-15

---

### 6. LINES (11 objects)
**Fill**: None
**Stroke**: Black, 0.39px width - `cls-5`

**Characteristics**:
- **Width**: 0.0001-0.0008 in (essentially zero-width paths)
- **Length**: 1.4-6.5 inches (LONG)
- **Aspect Ratio**: 0.00-15,637 (extreme!)
- **Anchors**: 3-5 (very simple)

**Easy Identifier**: **NO FILL + STROKE** ✅

**Classification Strategy**:
```javascript
if (!item.filled && item.stroked) {
    return "Line";
}
```

---

### 7. ARROWHEADS (22 objects)
**Fill**: Black - `cls-3`
**Stroke**: None

**Characteristics**:
- **Size**: 0.0008-0.0014 sq in (TINY triangles)
- **Dimensions**: ~0.045" × 0.02-0.03"
- **Aspect Ratio**: 1.45-2.78
- **Anchors**: Always 4 (triangles)

**Classification Strategy**:
- Very small area (<0.002 sq in)
- Exactly 4 anchors
- Aspect ratio 1.4-3.0

---

## Color-Based Classification

### CSS Classes in Use:
- **cls-1**: White fill (#fff)
- **cls-2**: Light gray fill (#cfcfcf) - **TAXIWAYS ONLY**
- **cls-3**: Black fill (evenodd) - **Everything else**
- **cls-4**: No fill (fill-rule: evenodd)
- **cls-5**: No fill + black stroke (0.39px) - **LINES ONLY**
- **cls-6**: No fill + black stroke (0.42px)
- **cls-7**: No fill + black stroke (0.39px, dashed)

---

## Revised Classification Strategy

### Phase 1: Easy Wins (Color/Stroke-Based)
1. **Taxiways**: Fill color = #cfcfcf (gray)
2. **Lines**: No fill + has stroke

### Phase 2: Geometric Discrimination (All use cls-3 black fill)
Need to differentiate:
- Text (682 objects, 2.7-17+ sq in, varied shapes)
- Footprints (28 objects, 0.003-15 sq in, mixed)
- Runways (2 objects, 6-7 sq in, low aspect)
- Arrowheads (22 objects, <0.002 sq in, 4 anchors)
- Stars (2 tiny, <0.005 sq in)

#### Size-Based Filtering:
```
< 0.002 sq in:    Arrowheads
< 0.01 sq in:     Stars (tiny) or small Footprint markers
0.01-1 sq in:     Small Footprints or small Text groups
1-5 sq in:        Medium Footprints or Text groups
5-10 sq in:       Large Footprints, Runways, or large Text
> 10 sq in:       Large Taxiways, Footprints, or Text blocks
```

#### Anchor Count Heuristic:
```
4 anchors + tiny:          Arrowheads or Runways
4-20 anchors + medium:     Footprints
10-80 anchors + large:     Text groups
10-100 anchors + gray:     Taxiways
```

---

## Key Insights for New Script

1. **Taxiways and Lines are EASY**: Use color/stroke detection
2. **Text is NOT individual letters**: Entire text blocks/groups
3. **Size ranges overlap significantly**: Can't use size alone
4. **Most objects share cls-3**: Need multi-factor scoring
5. **Arrowheads are distinct**: Tiny + exactly 4 anchors
6. **Your original dimensions (0.008" × 0.063") don't appear in this data!**

---

## Questions for User

1. Does the **unclassified** data have individual letters, or are they already grouped like this?
2. Are the tiny "Footprints" (0.003 sq in) actually building markers/symbols?
3. Do you want to classify "Stars" separately, or group with symbols?
4. Should small text groups (<1 sq in) be treated differently from large ones (>10 sq in)?

---

## Recommended Next Steps

1. **Run AnalyzeClassifiedDocument.jsx** on the actual AI file to get precise metrics
2. **Build a hybrid classifier**:
   - Rule-based for Taxiways (color) and Lines (stroke)
   - Geometric scoring for the cls-3 black-filled objects
3. **Add fill color detection** to ExtendScript classifier
4. **Adjust thresholds** based on actual data ranges above

---

## Optimal Thresholds (Based on Sample)

```javascript
CONFIG = {
    // Taxiways (by color)
    taxiwayFillRGB: [207, 207, 207],

    // Lines (by stroke)
    lineMaxArea: 0.01,  // Lines have near-zero area

    // Arrowheads
    arrowheadMaxArea: 0.002,
    arrowheadAnchorCount: 4,
    arrowheadAspectMin: 1.4,
    arrowheadAspectMax: 3.0,

    // Stars (tiny ones)
    starMaxArea: 0.01,
    starAnchorCount: 10,  // Often pentagon stars

    // Runways
    runwayMinArea: 5.0,
    runwayAspectRange: [0.3, 0.8],  // Or [1.25, 3.5] for horizontal
    runwayMaxAnchors: 6,

    // Footprints (buildings)
    footprintMinArea: 0.05,  // Exclude tiny markers
    footprintMaxArea: 20.0,
    footprintAspectRange: [0.4, 2.5],

    // Text (groups)
    textMinArea: 1.0,  // Smaller text might be footprint labels
    textAnchorDensityMin: 5,  // Anchors per square inch
}
```

---

**Next Action**: Should I create a new classifier based on this ACTUAL data structure?
