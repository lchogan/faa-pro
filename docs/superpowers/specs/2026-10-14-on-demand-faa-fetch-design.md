# On-Demand FAA Fetch — Design Spec

**Date:** 2026-10-14
**Branch:** `lchogan/faa-downloader-integration`
**Status:** Approved design, pending implementation plan

## Purpose

Today the classifier processes an airport PDF that must already exist in a
pre-built FAA download directory. That directory is produced in bulk,
periodically, by a separate project (`/Users/lukehogan/aoa-code/faa-downloader`).

This change makes the classifier fetch the **latest** FAA files on demand,
per run, given only an FAA LID code. Running the entry script on an airport —
new or already classified — always downloads fresh FAA data, classifies it,
and saves it in the long-term artwork folder. The standalone bulk downloader
is superseded and marked deprecated in this repo's README.

## Intended outcome (design brief)

The operator runs one script, is prompted for a destination (pre-filled
default) and one or more FAA LID codes, and the script:

1. Downloads the latest FAA data for each LID directly from the FAA.
2. Classifies each airport with the existing pipeline.
3. Saves each result in the destination, de-duplicating folder names so a
   re-run of an already-classified airport does not overwrite prior work
   (`ord`, then `ord-1`, `ord-2`, …).

The existing interactive **"Place approach lights?"** prompt is preserved.
There is **no** "Proceed?" confirmation and **no** source-folder prompt.

## Locked decisions

| Decision | Choice |
|----------|--------|
| Download scope | Full parity: APD diagram PDF + approach chart + NASR + macOS Finder color label |
| NASR per run | Re-download the full 28-day zip every run (no caching) |
| Runway data | Refresh the classifier's runway designations from the freshly-downloaded NASR (per-airport CSV via `--nasr-rwy`) |
| Save location | Artwork-folder default from `classify-wrapper.conf`, overridable at an interactive destination prompt |
| Entry point | Repurpose `classify-airports.sh` |
| Multi-code | Allow multiple comma-separated LID codes |
| Dedup | Numeric suffix on the folder name (`ord-1`, `ord-2`, …) |
| Lighting | Keep the interactive "Place approach lights?" prompt |
| Downloader logic | Port into this repo; deprecate the standalone project (this repo's README only; the separate repo is untouched) |

## Architecture

### Components

**1. `python/pipeline/faa_fetch.py`** — new reusable module. Ports the proven
logic from `faa-downloader/download_airports.py`:

- `get_current_cycle(today: datetime | None = None) -> tuple[str, str]` —
  AIRAC cycle string (`YYNN`) and effective date. `today` parameter added
  for unit-testing; defaults to `datetime.now(timezone.utc)`.
- `make_session(timeout) -> requests.Session` — retry/backoff session with a
  browser-like user agent.
- `fetch_metafile(cycle, session) -> ET.Element` — fetch the d-TPP metafile.
- `parse_tpp_records(root) -> (apd_map, iap_map)` — `{LID: apd_pdf_name}` and
  `{LID: [(chart_name, pdf_name), …]}`.
- `select_best_approach(iap_list, priority) -> (chart_name, pdf_name) | None`.
- `download_nasr(date_str, session) -> NasrData` — download the full 28-day
  zip; parse `APT_BASE.csv` + `CLS_ARSP.csv` for metadata and airspace class.
  Exposes the inner CSV zip so runway rows can be extracted.
- `extract_runway_rows(inner_zf, lid) -> list[dict]` — pull the target
  airport's `APT_RWY.csv` rows (needs only `ARPT_ID`, `RWY_ID`, enough to
  round-trip the columns `load_nasr_runways` reads).
- `determine_airspace_class(lid, row)`, `set_finder_label(folder, class, …)`,
  `download_pdf(url, dest, session, timeout)`, DMS/field helpers.
- `dedup_folder(parent: Path, lid: str) -> Path` — **pure** function: returns
  the first non-existing of `parent/lid`, `parent/lid-1`, `parent/lid-2`, …
- Config values become **named module constants**: `AIRSPACE_LABEL_COLORS`
  (airspace class → Finder label index), `APPROACH_PRIORITY`, timeouts. This
  removes the `config.json` dependency the old downloader carried.

**2. `python/fetch_airport.py`** — new thin CLI.

```
python fetch_airport.py --lid ORD --dest-parent <artwork-folder> [--priority ILS,RNAV,…]
```

Flow:
1. Compute cycle + effective date.
2. Fetch d-TPP metafile; resolve the APD pdf for the uppercased LID. If none,
   exit non-zero with a clear message ("no FAA airport diagram for <LID>").
3. Download NASR; resolve the metadata row + airspace class for the LID.
4. `folder = dedup_folder(dest_parent, lid.lower())`; `folder.mkdir(parents=True)`.
5. Set the macOS Finder color label on `folder`.
6. Download `<lid>-faa.pdf` (APD) and `<lid>-approach.pdf` (best approach;
   a failed approach download warns but does not abort the airport).
7. Extract the airport's runway rows and write `<lid>-nasr-rwy.csv` (same
   header columns `load_nasr_runways` consumes).
8. Print two machine-readable lines to stdout: `FOLDER=<abs path>` and
   `PDF=<abs path>`.

The PDF is always named by the **true LID** (`ord-faa.pdf`) even when the
folder is deduped (`ord-1`), so the classifier derives the correct airport
code and NASR key from the filename.

**3. `classify-airports.sh`** — rewritten entry flow:
1. Load the destination default from `classify-wrapper.conf` (hardcoded
   fallback retained).
2. Prompt **destination** `[default]` (Enter accepts).
3. Prompt **LID code(s)**, comma-separated; trim + lowercase.
4. Prompt **"Place approach lights? [y/N]"** → `export PLACE_LIGHTS`.
5. For each code: run `fetch_airport.py` with `--dest-parent <destination>`,
   capture `FOLDER` and `PDF` from stdout, `export NASR_RWY_CSV=<folder>/<lid>-nasr-rwy.csv`,
   then call `classify.sh "$PDF"`.

The source-folder prompt and the `mv`-from-dump logic are removed. The
"Proceed?" confirmation is removed. Uses the same venv python interpreter and
`PROJECT` resolution as `classify.sh`.

**4. `classify.sh`** — one addition inside `process_one`: if `NASR_RWY_CSV` is
set and the file exists, append `--nasr-rwy "$NASR_RWY_CSV"` to the
`classify_pipeline.py` invocation. The existing `PLACE_LIGHTS` /
`--place-lights` pass-through is untouched.

**5. `python/pyproject.toml`** — add `requests>=2.31` to `dependencies`.

**6. `README.md`** — document the on-demand flow; note the standalone
`faa-downloader` is superseded by `classify-airports.sh` + `fetch_airport.py`.

### Data flow

```
classify-airports.sh
  ├─ prompt: destination, LID(s), place-lights
  └─ per LID:
       fetch_airport.py --lid <LID> --dest-parent <dest>
         ├─ d-TPP metafile  → APD pdf name
         ├─ NASR zip        → airspace class (Finder label) + runway rows
         ├─ dedup_folder    → <dest>/<lid>[-N]
         ├─ download        → <lid>-faa.pdf, <lid>-approach.pdf
         └─ write           → <lid>-nasr-rwy.csv
       classify.sh <lid>-faa.pdf   (NASR_RWY_CSV + PLACE_LIGHTS exported)
         └─ existing pipeline → <lid>-diagram.svg / .ai
```

### Error handling

- **No APD for LID:** `fetch_airport.py` exits non-zero with a clear message;
  the wrapper skips that code and continues, exiting non-zero at the end if
  any code failed.
- **Network failures:** handled by the retry session; a hard failure aborts
  that one airport, not the batch.
- **Approach chart missing/failed:** warn, continue (the diagram is the
  required artifact).
- **Dedup:** never overwrites an existing folder.

### Testing

Extends the existing `python/tests/` pytest suite.

- `test_faa_fetch.py`:
  - `dedup_folder` — `tmp_path`: empty → `lid`; with `lid` present → `lid-1`;
    with `lid`, `lid-1` → `lid-2`.
  - `parse_tpp_records` — small d-TPP XML fixture with one APD + two IAP
    records → correct maps.
  - `select_best_approach` — priority ordering + fallback-to-first.
  - `get_current_cycle(today=…)` — known date(s), e.g. 2026-02-21 → `("2602", "2026-02-19")`.
  - `extract_runway_rows` — small `APT_RWY.csv` fixture (zipped) → correct
    rows for one LID, filtered from others.

Network functions (`fetch_metafile`, `download_nasr`, `download_pdf`) stay
thin and are **not** exercised against the live FAA; only the pure parsers
and helpers are unit-tested.

## Explicitly out of scope (YAGNI)

- No `config.json` port — named module constants instead.
- No NASR caching — full zip every run, per the locked decision.
- No threading — one LID at a time.
- No metadata/coords JSON — already moved to `ecom-dashboard/scripts/add_airport.py`.
- The separate `faa-downloader` repo is not modified (only referenced as
  deprecated from this repo's README).

## Files touched

| File | Change |
|------|--------|
| `python/pipeline/faa_fetch.py` | new module |
| `python/fetch_airport.py` | new CLI |
| `classify-airports.sh` | rewritten entry flow |
| `classify.sh` | add `NASR_RWY_CSV` → `--nasr-rwy` pass-through |
| `python/pyproject.toml` | add `requests` dependency |
| `python/tests/test_faa_fetch.py` | new tests + fixtures |
| `README.md` | document new flow; deprecate bulk downloader |
