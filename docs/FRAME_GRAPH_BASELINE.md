# Frame Graph Baseline

The first runnable `rest_app` implementation is a direct-score frame-graph baseline.

## Scope
This baseline supports:
- synchronized multi-camera frame-folder input
- Ultralytics detection per camera image
- node construction from detection boxes
- cross-camera graph construction for each timestamp
- direct edge scoring with selected spatial edge features
- connected-component clustering after score thresholding
- simple temporal ID carry-over through same-camera IoU matching against the previous frame clusters
- MOT export
- annotated frame export

## Why this baseline exists
The V2 plan requires a usable first pipeline before the full configurable GNN stack is implemented. This baseline provides:
- a real graph-construction path
- a real sink/output path
- a first end-to-end execution route for `rest_app`
- a concrete object model for upcoming notebook experiments

## Current limitations
- direct-score only
- no GNN or hybrid reasoning yet
- no DGL yet
- no tracklet graph yet
- no segmentation or attribute enrichment wired into node or edge scoring yet
- temporal continuity uses only a simple same-camera IoU carry-over heuristic

## Expected dataset layout
The current implementation expects a dataset root with one subfolder per camera, each containing synchronously ordered images.

Example:

```text
some_dataset/
  cam0/
    000001.jpg
    000002.jpg
  cam1/
    000001.jpg
    000002.jpg
  cam2/
    000001.jpg
    000002.jpg
```

If the filenames do not match across cameras, the implementation falls back to the frame index as the timestamp label.
