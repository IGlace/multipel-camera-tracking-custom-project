# Realtime Tracklet Runtime Baseline

The first runnable tracklet-mode `realtime_app` implementation uses one Ultralytics local tracker per camera, builds rolling tracklet summaries, and performs cross-camera global association on those active tracklets.

## Supported inputs
- synchronized multi-camera frame folders
- a single video file
- a folder of videos (treated as one video source per camera)

## Scope
This baseline supports:
- one Ultralytics local tracker instance per camera
- normalized local tracker outputs
- rolling tracklet summaries with a configurable window size
- cross-camera graph construction over active tracklets
- direct edge scoring on tracklet summaries
- connected-component clustering after score thresholding
- temporal global-ID carry-over across consecutive tracklet graphs
- MOT export
- annotated frame export
- optional annotated video export
- optional live grid display
- optional graph-debug figure export

## Current limitations
- local tracker correction and retroactive reconciliation are not implemented yet
- no GNN reasoning yet
- no segmentation or attribute enrichment in scoring yet
- no finalized long-horizon tracklet management yet beyond the rolling window
