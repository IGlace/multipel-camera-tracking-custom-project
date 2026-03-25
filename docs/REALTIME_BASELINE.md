# Realtime Frame Runtime Baseline

The first runnable `realtime_app` implementation is a direct-score frame-runtime baseline.

## Supported inputs
- synchronized multi-camera frame folders
- a single video file
- a folder of videos (treated as one video source per camera)

## Scope
This baseline supports:
- Ultralytics detection per camera image or video frame
- node construction from detection boxes
- cross-camera graph construction for each timestamp
- direct edge scoring with selected spatial edge features
- connected-component clustering after score thresholding
- simple temporal ID carry-over through same-camera IoU matching against the previous frame clusters
- MOT export
- annotated frame export
- optional annotated video export
- optional live grid display
- optional graph-debug figure export

## Why this baseline exists
The V2 plan requires `realtime_app` to remain distinct from the research-oriented offline apps while still following the same overall reasoning for frame-mode operation. This baseline provides:
- a true runtime-oriented input layer
- shared output sinks across apps
- a first live-display capable execution route

## Current limitations
- frame-mode only
- no local tracker usage yet
- no tracklet runtime yet
- no reconciliation policy yet beyond simple temporal carry-over
- no segmentation or attribute enrichment in scoring yet
