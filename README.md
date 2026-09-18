# FUMD-AI Preprocessing Workflow

Turns raw SUMO (vehicle mobility) and OMNeT++ (cellular network) simulation
output into a labeled, AI-ready dataset for training cellular
handover / migration prediction models.

## Quick start: run the whole pipeline in one call

`run_pipeline.ipynb` (repository root) is a master notebook that calls
every step below in order via [papermill](https://papermill.readthedocs.io/),
chaining each step's output into the next. Edit its single **parameters
cell** (or override headlessly) and run it - no need to open the seven
step notebooks individually. Out of the box it runs against the bundled
`example-data/` with no edits at all. Every intermediate/final file, plus
a fully executed copy of each step notebook (for provenance), is written
to `OUTPUT_DIR` (default: `pipeline_run/`).

A full run needs all three true raw inputs: the SUMO fcd-output XML, the
SUMO<->VEINS/OMNeT id mapping file, and the OMNeT++ network-metrics data
- either as an already-extracted `RAW_OMNET_PATH` CSV (the common case,
since `RUN_OMNET_EXTRACT` defaults to `False`):

```
papermill run_pipeline.ipynb out.ipynb \
    -p RAW_SUMO_XML_PATH my_run_fcd.xml \
    -p RAW_OMNET_PATH my_run_omnet_export.csv \
    -p MAPPING_PATH my_mapping.txt
```

or, if you only have the raw `.vec` file, by pointing at that instead and
turning on `RUN_OMNET_EXTRACT`. Step 1 offers two interchangeable ways to
extract it - `OMNET_EXTRACTOR_METHOD = "extractvectors"` (default, needs
the external tool - see Requirements below) or `"python"` (dependency-free,
built into the notebook, slower on very large files - see Step 1's own
notebook for a detailed comparison):

```
papermill run_pipeline.ipynb out.ipynb \
    -p RAW_SUMO_XML_PATH my_run_fcd.xml \
    -p RAW_OMNET_VEC_PATH my_run_vector.vec -p RUN_OMNET_EXTRACT True \
    -p OMNET_EXTRACTOR_METHOD python \
    -p MAPPING_PATH my_mapping.txt
```

See the notebook itself for the full parameter list and what each one
does; the individual step notebooks below remain fully usable on their
own for running/testing/debugging one step at a time.

## Pipeline overview

Seven notebooks, numbered in run order. Six form the core chain from raw
simulator output to the AI-ready dataset; one is optional (diagnostics/figures):

| # | Notebook | Purpose | Input (default) | Output (default) |
|---|----------|---------|-------|--------|
| 1 | `step_1_parse_raw_sumo_and_omnet.ipynb` | Turn the two genuinely raw simulator outputs into flat CSVs: parse a SUMO fcd-output XML into a per-vehicle-per-timestep trajectory CSV, and/or extract named vectors (SINR, CQI, RLC delay/throughput, serving cell, ...) from a raw OMNeT++ `.vec` file into a long-format CSV, via either the external `extractvectors` tool or a bundled dependency-free pure-Python extractor (`OMNET_EXTRACTOR_METHOD`). | `example-data/raw_sumo_fcd.xml`, `example-data/raw_omnet_vector.vec` | `sumo_trajectory_base.csv`, `omnet_export.csv` |
| 2 | `step_2_add_past_position_columns.ipynb` | Add lagged `x-1..x-7`/`y-1..y-7` past-position columns to the Step 1 SUMO CSV (SUMO's own `999999` "no history yet" sentinel is left as-is; Step 5 replaces it later). | `sumo_trajectory_base.csv` (Step 1 output) | `sumo_trajectory.csv` |
| 3 | `step_3_generate_omnet_matrix.ipynb` | Clean the raw OMNeT++ vector export and pivot it into a wide, per-vehicle-per-timestep table of network metrics, with lagged/lead serving-cell columns. | `example-data/raw_omnet_export.csv` (bundled - see note below) | `omnet_feature_matrix.csv` |
| 4 | `step_4_merge_sumo_omnet.ipynb` | Align SUMO and OMNeT++ vehicle ids and merge the SUMO trajectory (position, speed, lane, ...) with the Step 3 network-metric matrix. | `sumo_trajectory.csv`, `omnet_feature_matrix.csv`, `example-data/sumo_veins_mapping.txt` | `combined_dataset.csv` |
| 5 | `step_5_fix_past_positions.ipynb` | Replace the `999999` sentinel SUMO uses for "no history yet" in the lagged position columns with each vehicle's first known coordinate. | `combined_dataset.csv` (Step 4 output) | `combined_dataset_fixed.csv` |
| 6 | `step_6_label_cell_migrations.ipynb` | Reconstruct each vehicle's stable serving-cell history and label every row with `migration` (0/1/2, pre-handover warning window) and `destination` (the cell the vehicle will actually settle on). **This is the AI-ready dataset.** | `combined_dataset_fixed.csv` (Step 5 output) | `dataset_labeled_w<W>.csv` (+ event logs) |
| 7 *(optional)* | `step_7_visualize_migrations_optional.ipynb` | Diagnostics and figures: time spent in unstable mobility, event-type distribution, events per vehicle, spatial map of handovers. Not required to produce the dataset. | Step 6 outputs | plots + summary tables |

The true raw inputs to this workflow are a SUMO fcd-output XML file, a raw
OMNeT++ `.vec` vector file, and a SUMO<->VEINS/OMNeT id mapping file
(`SUMO=<sumo_id> VEINS=<veins_id>` lines). The mapping file is produced
directly by the OMNeT++/VEINS simulation itself and is treated as a given
input to this workflow, not something generated by it. Steps 1/2 turn the
two raw simulator outputs into the flat CSVs Steps 3-6 read; Step 7 is
optional/exploratory and not required to produce the AI-ready dataset.

**Note on the mapping file's id reuse.** A SUMO id can legitimately appear
more than once (SUMO teleporting a stuck vehicle to a new position hands
it a new VEINS module slot each time), and so can a VEINS/OMNeT id
(reassigned to a different vehicle once its previous occupant's slot is
freed). Step 4 resolves this per-timestep rather than requiring a strict
1:1 mapping - see v1.1.3 in Development notes.

**Note on Step 3's default input.** Step 1's default extractor,
`extract_omnet_vectors()`, needs the external `extractvectors` tool, so
Step 3's own default `RAW_OMNET_PATH` points at the bundled, ready-made
`example-data/raw_omnet_export.csv` rather than chaining onto Step 1's
output - this keeps Steps 3 onward runnable without that external tool
(or Step 1's pure-Python alternative, `OMNET_EXTRACTOR_METHOD = "python"`,
which needs nothing extra but is not the default). Point Step 3 at Step
1's actual `omnet_export.csv` once you have real extracted data, from
either extractor.

## Example data

`example-data/` bundles a tiny, synthetic (not real simulation) dataset -
2 vehicles over 8 simulated seconds, with one genuine handover - purely so
every notebook is runnable immediately, using nothing but its own default
parameters, straight after downloading this workflow. It is deliberately
not real/representative simulation data (see the Validation section below
for that); it exists only to prove the pipeline mechanics work and to give
reviewers something to run without needing multi-gigabyte inputs. Replace
every path in each notebook's `parameters` cell with your own simulation's
files before drawing any real conclusions.

## Repository layout

```
.
├── README.md
├── CITATION.cff                 citation metadata (GitHub/Zenodo citation widget)
├── run_pipeline.ipynb            master notebook: runs Steps 1-7 in one call
├── LICENSE.txt                  MIT (source code)
├── LICENSE-CC-BY-4.0.txt        CC BY 4.0 (explanatory text/figures)
├── requirements.txt
├── ro-crate-metadata.json       FAIR/WorkflowHub packaging metadata
├── example-data/                tiny bundled synthetic example (see below)
│   ├── raw_sumo_fcd.xml
│   ├── raw_omnet_vector.vec
│   ├── raw_omnet_export.csv
│   └── sumo_veins_mapping.txt
├── src/
│   └── fumd_workflow/
│       ├── __init__.py
│       └── labeling.py          shared trajectory + migration-labeling logic
└── notebooks/
    ├── step_1_parse_raw_sumo_and_omnet.ipynb
    ├── step_2_add_past_position_columns.ipynb
    ├── step_3_generate_omnet_matrix.ipynb
    ├── step_4_merge_sumo_omnet.ipynb
    ├── step_5_fix_past_positions.ipynb
    ├── step_6_label_cell_migrations.ipynb
    └── step_7_visualize_migrations_optional.ipynb
```

Notebooks 1, 2, 3, 4, 5 and 7 are self-contained. Notebook 6 imports
`build_trajectories` and `annotate_migrations` from `src/fumd_workflow`
rather than redefining that logic inline - see that module's docstrings
for a full description of how handovers are classified (normal handover,
delayed handover, ping-pong, etc).

## Requirements

- Python >= 3.10
- a `sed` binary on `PATH` (used in Step 3 for fast cleaning of
  multi-gigabyte raw OMNeT++ exports); works with both GNU sed (default on
  Linux) and the BSD sed shipped by default on macOS - no separate install
  needed on either platform
- Python packages: see `requirements.txt` (includes `papermill`, required
  by `run_pipeline.ipynb`)

```
pip install -r requirements.txt
```

### Tested with

**Simulation environment** used to generate the real SUMO/OMNeT++ data this
workflow was validated against (see Validation below):

| Tool | Version | Link |
|------|---------|------|
| OMNeT++ | 6.3.0 | https://omnetpp.org/download/old |
| Simu5G | 1.4.4 | https://github.com/Unipisa/Simu5G/releases/tag/v1.4.4 |
| INET | 4.5.4 | https://inet.omnetpp.org/2024-10-29-INET-4.5.4-released.html |
| Veins | 5.3.1 | https://veins.car2x.org/download/ |
| SUMO | 1.22.0 | https://sumo.dlr.de/releases/1.22.0/ |

fcd-output from SUMO 1.27.1 has also been used successfully (see
Validation below) - the fcd XML format read by Step 1 has been stable
across this range.

`extractvectors` (from the [netperfmeter](https://github.com/dreibh/netperfmeter)
project, used by Step 1's default extractor, `extract_omnet_vectors()`)
was built from source and validated against real OMNeT++ output; it is
part of the INET 4.5.4 toolchain above. Its exact output format (extra
event/object-id columns, header style) has been observed to vary across
builds - Step 3's cleaning step (see Development notes) normalizes the
variations seen so far automatically, so this shouldn't require any
manual handling on your part. If you don't have `extractvectors`
installed, Step 1's `OMNET_EXTRACTOR_METHOD = "python"` option reads the
raw `.vec` file directly with no external tool at all - cross-checked
against real `extractvectors` output on the same data (see Step 1's own
notebook for the comparison); slower on very large files, but not
dramatically so.

**Python/notebook environment**, from a real end-to-end `papermill` run of
`run_pipeline.ipynb` on macOS with its default BSD `sed`:

| Package | Version |
|---------|---------|
| Python | 3.11 |
| pandas | 2.1.4 |
| numpy | 1.26.4 |
| jupyter | 7.0.8 |
| nbformat | 5.9.2 |
| papermill | 2.7.0 |
| sed | BSD sed (macOS default) |

## Running individual steps

`run_pipeline.ipynb` (see Quick start above) is the easiest way to run the
whole thing, but every step notebook also stands on its own. Each has a
single **parameters cell** (tagged `parameters`, near the top) listing
every input/output path and tunable value for that step. Run from the
repository root (so that Step 6's `sys.path.insert(0, "src")` finds the
shared module, and the `example-data/...` default paths resolve). Out of
the box, every notebook's defaults chain onto the previous step's default
output and onto the bundled `example-data/` files, so running 1 through 6
in order - with no edits at all - works immediately against the tiny
bundled example (7 is optional). For your own data, edit each notebook's
parameters cell to point at your own files instead.

Because every parameters cell is tagged, each notebook can also be run
headlessly and parameterized with
[papermill](https://papermill.readthedocs.io/) individually, e.g.:

```
papermill notebooks/step_3_generate_omnet_matrix.ipynb out_step3.ipynb \
    -p RAW_OMNET_PATH my_run_omnet_export.csv \
    -p OUTPUT_MATRIX_PATH my_run_omnet_feature_matrix.csv
```

## Validation

All seven notebooks were run end-to-end against real SUMO + OMNeT++
simulation data, from raw simulator output through to the final labeled
dataset, confirming correct output at every step. All 7 notebooks (and
`run_pipeline.ipynb` itself) were also validated top-to-bottom using
nothing but their own default parameters against the bundled
`example-data/`, confirming the whole chain is runnable immediately after
downloading the workflow.

The pipeline was additionally validated on a much larger real simulation
run (893 vehicles, a 43M-row raw OMNeT++ export, 1800 simulated seconds),
producing a final labeled dataset of ~9.15M rows with no missing values.
That run surfaced and fixed several edge cases this workflow now handles
correctly: `extractvectors` output format variations across builds, a
dual-connectivity network's redundant LTE-stack vectors, vehicles with no
downlink traffic at all, and SUMO/OMNeT id reuse in the vehicle mapping
file.

A second independent real simulation run (900 vehicles, 1800 simulated
seconds, generated with SUMO 1.27.1) was used to validate v1.1.0's new
extractor: Step 1's `OMNET_EXTRACTOR_METHOD = "python"` option processed a
28 GB / 835M-line raw `.vec` file end-to-end (its largest input to date)
via `run_pipeline.ipynb`, and the full pipeline completed with zero errors
and zero missing values in the final labeled dataset (89,757 rows across
all 900 vehicles - this run's SUMO fcd-output happened to be recorded at
1 Hz rather than the finer rate used in the run above, hence the smaller
row count relative to OMNeT's own resolution; all 9 base stations are
still represented and 2,048 handover events were detected).

The v1.1.1 crash fixed below (see Development notes) was found separately,
running the workflow against a third real dataset on a different machine
(Python 3.14, pandas 3.0.5).

A fourth real simulation run (`VoipDl-Urban-1200_3`, 1200 vehicles, 1800
simulated seconds, a ~35.5 GB raw `.vec` file - the largest input to date)
was used to confirm the v1.1.3 and v1.1.4 fixes below: the pipeline
completed end-to-end with zero errors via `run_pipeline.ipynb`, correctly
resolved all 22 teleportation-affected mapping rows (SUMO ids reused
across multiple OMNeT ids) with every remapped id matching its OMNeT
counterpart, and produced a labeled dataset (124,392 combined rows, 5,448
vehicle runs, 2,777 actionable migration events) with the corrected event
case labels (`C2b_pingpong`: 169, `C3_handover_sin_historico`: 48) and no
leftover old-name cases. This dataset's SUMO fcd-output is logged at 1 Hz
against OMNeT's much finer resolution, so Step 4's exact-time merge
naturally keeps only ~1% of OMNeT samples (those landing on a whole
second) - the same expected behavior already noted for the second
validation run above, not a defect.

## Development notes

**v1.1.5** removed the last two `papermill` "Unable to parse line" /
"Passed unknown parameter" warnings, a cosmetic issue from the same
fragile parameter-cell line-parser behind the `WINDOW_S_VALUES` CLI bug
documented under Validation/earlier notes. Two trailing comments were
triggering it: Step 2's `ROWS_PER_SECOND` line had an embedded `=` inside
its parenthetical example, and Step 6's `TOL_S` line had embedded quotes
and a `>=` comparison. Reworded both comments to avoid embedded `=` and
quote characters (matching `run_pipeline.ipynb`'s own `TOL_S` line, which
never triggered the warning). No logic changed - confirmed via a real
rerun of `VoipDl-Urban-1200_3` producing byte-for-byte identical output to
the pre-fix run, with the warnings no longer appearing anywhere in the
papermill log.

**v1.1.4** corrected two swapped case codes in Step 6's handover
classification (`src/fumd_workflow/labeling.py`), used by Step 7's
event-type-distribution plot:

- The case for a vehicle's first-ever stable run, reached with no earlier
  stable "from" cell to compare against (no prior history), was labeled
  `C2b`. It is now `C3`.
- The ping-pong case (stable -> short runs only -> back to the *same*
  stable cell, not a real handover) was labeled `C3`. It is now `C2b`.
- Updated everywhere the codes appear: `labeling.py`'s docstring table and
  both `_log_case()` call sites, Step 7's `EVENT_ORDER`/`EVENT_LABELS`, and
  Step 6's own descriptive comment (which previously grouped C2a/C2b
  together as "delayed handovers" - no longer accurate now that C2b is
  ping-pong, not a delayed handover at all).
- Verified with a synthetic reproduction of both cases run through the
  actual `annotate_migrations()` function: a no-prior-history vehicle now
  tags as `C3_handover_sin_historico` and a ping-pong vehicle now tags as
  `C2b_pingpong`, and Step 7's relabeled `EVENT_ORDER` correctly lines up
  against that output.

**v1.1.3** fixed a silent data-loss bug in Step 4's SUMO<->OMNeT id
alignment, found while investigating what looked like mapping-file
corruption on a fourth real dataset (`VoipDl-Urban-1200_3`) - a SUMO id
appeared 11 times in a row, each paired with a different OMNeT id. That
turned out to be legitimate: SUMO's *teleportation* feature (relocating a
vehicle stuck in gridlock rather than removing it) makes VEINS assign the
relocated vehicle a new `car[]` module slot, so the same SUMO id gets a
new mapping row each time it teleports. The mirror case - a single OMNeT
id reused by different SUMO ids after VEINS frees and reassigns a module
slot with no teleport involved - was already known from `VoipDl-Urban-900_1`
(see v1.0.2 below).

- The existing fix only handled the second cause: it collapsed every
  repeated SUMO id onto the *first* OMNeT id it was ever paired with. On
  the teleportation case this ran without error but was wrong - it
  silently mis-attributed a teleported vehicle's entire trajectory
  (before *and* after every teleport hop) to its pre-teleport OMNeT id,
  and Step 4's own merge then dropped the other hops' feature-matrix rows
  as "unmatched" with no warning at all. On a mapping file where *both*
  causes are present (900_1) it could instead raise `ValueError:
  Replacement lists must match in length`.
- Replaced the collapse-to-a-single-id approach with per-timestep
  resolution: Step 3's feature matrix already records when each OMNeT id
  starts being logged, so for every `(SUMO id, time)` in the trajectory we
  pick whichever of that SUMO id's candidate OMNeT ids had already started
  by that timestep (`pandas.merge_asof`, matched within each SUMO id).
  This handles teleportation (a vehicle's post-teleport rows resolve to
  its new OMNeT id automatically, so each teleport hop is treated as a
  distinct vehicle for migration-labeling purposes, rather than injecting
  a fake instantaneous jump) and module-slot reuse (two different SUMO
  vehicles sharing one OMNeT id at different times both resolve correctly)
  uniformly, with no assumption that either id repeats at most once, and
  without needing the mapping table to be a strict bijection at all. This
  is not an opt-in flag - it replaces the old logic outright, since the
  old behavior was a silent correctness bug rather than a reasonable
  default.
- Verified with synthetic reproductions of both cases (a teleporting SUMO
  id and a reused OMNeT id) run through the actual generated Step 4
  notebook: both split/resolve correctly and no feature-matrix rows are
  lost. Confirmed against `VoipDl-Urban-1200_3`'s own full pipeline run -
  see Validation above.

**v1.1.2** generalized Step 3's "vehicle with a metric never recorded"
handling, after the v1.1.1 retest surfaced the same class of `AssertionError:
unexpected NaNs remain after fill` on a different dataset - this time for a
metric outside the fixed `NO_TRAFFIC_FILL_ZERO_VECTORS` list added in v1.0.2
(that list only covered the three RLC downlink metrics that happened to be
missing on the dataset used to write it). Rather than extend the list again
for whichever specific metric this next dataset was missing, the fix removes
the fixed list entirely: any metric still `NaN` after per-vehicle
ffill/bfill - meaning it was never recorded even once for that vehicle,
since ffill/bfill can only propagate a value that exists somewhere in that
vehicle's own timeline - is now filled with `0` regardless of which column
it is, consistent with `servingCell`'s own established "not yet connected"
convention elsewhere in this data. A summary of how many values got filled,
and in which columns, is printed so an occasional quiet vehicle (expected)
stays distinguishable from a sudden large jump (a sign something upstream,
like the id mapping, is actually broken). Pending confirmation on the
machine/dataset that originally hit this.

**v1.1.1** fixed a pandas-version-sensitive crash in Step 6's labeling
logic (`src/fumd_workflow/labeling.py`), found when running the workflow
on a third machine (Python 3.14, pandas 3.0.5 - newer than any
environment this project had previously been tested against):

- `annotate_migrations()` marks every row inside a pre-handover warning
  window with `migration=2`, then overwrites just the first row of that
  window with `migration=1` to flag the actual onset. That single-cell
  overwrite used `out.at[win_idx[0], "migration"] = pd.Series(1,
  dtype="Int8")` - passing a length-1 `pd.Series` where `.at[]` expects a
  bare scalar. Pandas versions used elsewhere in this project's testing
  (2.1.4 and 2.3.3) silently unwrap a length-1 Series into its scalar
  value there, so this went unnoticed; pandas 3.0 made Copy-on-Write
  mandatory and removed a number of legacy lenient-coercion fallbacks in
  `.at[]`/masked-array `__setitem__`, so 3.0.5 rejects it instead, raising
  a chained `TypeError` ("only 0-dimensional arrays can be converted to
  Python scalars") followed by two `ValueError`s ending in "Incompatible
  indexer with Series". Fixed by assigning the plain scalar `1` directly,
  which is correct and has been verified to work identically across
  pandas versions old and new - it never relied on the lenient behavior to
  begin with.
- Audited the rest of `labeling.py` and every notebook for the same class
  of issue (`.at[]`/`.iat[]` misuse, chained indexing assignment,
  deprecated/removed pandas APIs): this was the only occurrence anywhere
  in the codebase.

**v1.1.0** added an opt-in vector-extraction alternative and fixed a
visualization bug, both re-validated end-to-end against the same real
`VoipDl-Urban-900_1` dataset described in Validation above after making
these changes:

- Step 1 gained a second, opt-in vector-extraction implementation:
  `extract_omnet_vectors_pure_python()`, selected via the new
  `OMNET_EXTRACTOR_METHOD` parameter (`"extractvectors"`, the existing
  default, or `"python"`). It reads a raw `.vec` file directly with no
  external tool, cross-checked against real `extractvectors` output on
  the full `VoipDl-Urban-900_1` dataset: identical Time/Object/Value
  content for every spot-checked vector, ~26% fewer output rows (it
  restricts extraction to `car[...]` modules and skips the known
  constant-0 LTE-stack `servingCell` duplicate at the source, rather than
  relying on Step 3's cleaning to remove that noise downstream), and full
  floating-point precision preserved from the source file (`extractvectors`
  truncates to 6 decimals). Not dramatically slower despite being pure
  Python: a 500 MB / 16.9M-line slice processed in ~5 seconds in testing,
  extrapolating to roughly 4-5 minutes for the full 26 GB file. See Step
  1's own notebook (Part 2b) for the full comparison.
- Step 7's event-type-distribution legend passed plain strings positionally
  as `legend()`'s handles alongside a `labels=` keyword, which raised a
  "mixed positional and keyword arguments" `UserWarning` and produced a
  broken legend. Fixed to build proper `Patch` proxy handles. Separately,
  its `bbox_to_anchor` y-coordinate was set above `1.0` (figure-fraction
  coordinates), which places a legend above the figure canvas entirely
  rather than just above the axes - a legend that wraps onto more than one
  row in that position gets silently clipped, leaving only the row closest
  to the canvas edge visible. Fixed by anchoring at `y <= 1.0` and
  reserving the vertical space it needs via `tight_layout(rect=...)`
  instead, plus a wider minimum figure size so the 2-column legend has
  room even with a single `WINDOW_S_VALUES` entry.

**v1.0.2** fixed bugs surfaced by the larger real run described in
Validation above:

- Step 3's LTE/NR dual-connectivity `servingCell` deduplication only
  matched `car[N].cellularNic.phy` immediately after a tab, which never
  fires when the module path carries a network-instance prefix (e.g.
  `NRSeveralBSALC.car[3]...`). Both the constant-0 LTE copy and the real
  NR copy were surviving into the pivot and getting collapsed via
  `aggregate("min")`, corrupting `servingCell` toward 0. Fixed to tolerate
  an arbitrary prefix before `car[`.
- Step 3's cleaning now normalizes `extractvectors` output format
  variations across builds: an extra event number and internal numeric
  object id before Time/Object, a space-separated (rather than
  tab-separated) header, and the vector name bundled with its format
  string and an empty Split field in one column.
- Step 3's raw-file header skip (via a Python file iterator, then handing
  that same file object to `sed` as a subprocess) corrupted the first data
  row - Python's internal read-ahead buffering silently desyncs the
  underlying file descriptor position. Fixed by piping through
  `tail -n +2 | sed` instead.
- Step 3 now reduces `Object` to a bare integer id (stripping the OMNeT++
  network's own top-level module name), which it never did before - this
  happened to work by coincidence when no such prefix existed, but fails
  Step 4's merge otherwise (a dtype mismatch between `int64` and `object`).
- Step 3 now fills certain RLC downlink metrics with 0 for vehicles with no
  downlink application traffic at all, rather than treating their absence
  as an unconditional error - they're genuinely never recorded for such
  vehicles, not missing data.
- Step 3's `rlcPduThroughputDl` moved to the always-zero metrics list:
  every recorded value for it is exactly 0 across a full real run, even
  for vehicles with real nonzero values on the near-identical
  `rlcThroughputDl`.
- Step 3's always-zero metrics were only being dropped in the resampling
  step, after the NaN check in the pivot step had already run - so a
  column about to be dropped anyway could still fail that check. Now
  dropped immediately after pivoting.
- Step 4's duplicate-SUMO-id remap built two independently-deduplicated
  arrays of ids and paired them up positionally, which only works when
  every reused SUMO id is reused exactly once. On a larger run this
  produced arrays of different lengths and could silently mis-pair ids
  even when lengths matched by coincidence. Fixed to build an explicit
  id-to-id mapping instead, correct regardless of how many times an id
  repeats.

**v1.0.1** fixed three bugs:

- Step 3's raw-export cleaning `sed` pattern used a `\t` escape that only
  GNU sed interprets as a tab character; BSD sed (macOS default) matched
  it as the literal letter "t" instead, silently corrupting the cleaned
  file rather than erroring. Fixed by embedding a literal tab byte in the
  pattern instead - identical behaviour on GNU and BSD sed.
- Step 1's (currently unused-by-default) `extract_omnet_vectors()` had
  the same class of bug in two `sed -i` calls with no backup-suffix
  argument - valid under GNU sed's "-i means no backup", but BSD sed
  requires an explicit suffix and would misparse the same invocation.
  Replaced both with plain Python read/replace/write.
- Step 3's resampling step passed `include_groups=False` to
  `.groupby().resample()`; that argument is only recognized by pandas
  >= 2.2 and raises `TypeError` on 2.1.4. Replaced with
  `.groupby(["Object", pd.Grouper(freq=...)])`, which behaves identically
  across pandas versions.
- Every notebook cell now carries a unique nbformat `id` field, fixing a
  `MissingIDFieldWarning` (transparently patched by nbformat today, but a
  hard error in a future nbformat release per the warning text itself).

The core migration-labeling logic lives as a single, tested version in
`src/fumd_workflow/labeling.py`.

## Authors

- Cristina Bernad, Miguel Hernandez University ([ORCID: 0000-0001-9537-415X](https://orcid.org/0000-0001-9537-415X))
- Sonja Filiposka <sonja.filiposka@finki.ukim.mk>, Ss. Cyril and Methodius University in Skopje ([ORCID: 0000-0003-0034-2855](https://orcid.org/0000-0003-0034-2855))
- Katja Gilly, Miguel Hernandez University ([ORCID: 0000-0002-8985-0639](https://orcid.org/0000-0002-8985-0639))

## Licence

Unless otherwise indicated:

* Source code in this notebook is licensed under the MIT License.

* Explanatory text and original figures are licensed under Creative
  Commons Attribution 4.0 International (CC BY 4.0). Input datasets
  retain the licences stated in their corresponding metadata or
  source records.

SPDX-License-Identifier: MIT

See `LICENSE.txt` (MIT, code) and `LICENSE-CC-BY-4.0.txt` (CC BY 4.0, text/figures).

## Acknowledgement

This work has been funded by the FUMD-AI project, an EOSC GRAVITY - Inter
Project with Grant Number 25-EOSC-GRV-INTER-013.

We gratefully acknowledge Polish high-performance computing infrastructure
PLGrid (HPC Centers: ACK Cyfronet AGH) for providing computer facilities and
support within computational grant no. PLGINT/2026/019844.

The research work was supported by the Open Science Cloud research laboratory
(OSC-LAB) at the Faculty of Computer Science and Engineering (FINKI), Ss.
Cyril and Methodius University in Skopje, North Macedonia.

## Citation

Please cite this workflow if you use it. See `CITATION.cff` and
`ro-crate-metadata.json` for structured citation/author metadata - a DOI
slot is reserved in both, to be filled in once the workflow is registered
on WorkflowHub.
