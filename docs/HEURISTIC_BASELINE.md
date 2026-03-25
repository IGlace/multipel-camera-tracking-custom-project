# Heuristic Frame-Match Baseline

The first runnable `heuristic_app` implementation is a non-graph direct matching baseline.

## Scope
This baseline supports:
- synchronized multi-camera frame-folder input
- Ultralytics detection per camera image
- node construction from detection boxes
- direct pairwise cross-camera scoring without graph libraries
- union-find grouping of detections after thresholding
- simple temporal ID carry-over through same-camera IoU matching against the previous frame clusters
- MOT export
- annotated frame export
- optional annotated video export

## Why this baseline exists
The V2 plan requires `heuristic_app` to remain separate from graph reasoning while keeping a similar conceptual flow. This baseline provides:
- a true non-graph association path
- a comparable inference output to the first `rest_app` baseline
- a concrete starting point for later heuristic experiments and notebook validation

## Current limitations
- frame-based only
- no tracklet matching yet
- no Hungarian or multi-stage matching yet
- no attribute- or segmentation-enriched scoring yet
- temporal continuity uses only a simple same-camera IoU carry-over heuristic

## Expected dataset layout
The current implementation expects a dataset root with one subfolder per camera, each containing synchronously ordered images.
