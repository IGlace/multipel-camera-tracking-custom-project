# Multiple Camera Tracking Custom Project

A unified GPU-first multi-camera multi-target tracking platform.

## Current status
This branch contains the Phase 1 foundation scaffold for the V2 architecture.

Implemented foundation pieces:
- single Dockerfile strategy
- shared runtime entrypoint
- GPU fail-fast bootstrap
- shared config package scaffold
- shared logging package scaffold
- app and trainer entrypoint scaffolds
- first typed interfaces for detector, segmentor, tracker, ReID, and attributes
- experiment notebook layer bootstrap
- first runnable `rest_app` frame-graph direct-score baseline
- first runnable `heuristic_app` frame-match baseline
- first runnable `realtime_app` frame-runtime baseline for frame folders, a single video file, or a folder of videos
- first runnable `realtime_app` tracklet-runtime baseline with Ultralytics local tracking and rolling tracklet summaries
- shared MOT, annotated frame, annotated video, live grid, and graph-debug sinks

## Planned top-level modules
- `mcmt_core/`
- `apps/rest_app/`
- `apps/heuristic_app/`
- `apps/realtime_app/`
- `trainers/reid_mgn/`
- `trainers/attributes/`
- `configs/`
- `tests_notebooks/`
- `outputs/`

## Execution model
This platform is designed to run through a single runtime entrypoint:

```bash
python scripts/entrypoint.py rest -- --config configs/rest/base.yaml --mode infer
python scripts/entrypoint.py heuristic -- --config configs/heuristic/base.yaml --mode infer
python scripts/entrypoint.py realtime -- --config configs/realtime/base.yaml --mode infer
python scripts/entrypoint.py trainer-reid-mgn -- --config configs/trainers/reid_mgn.yaml
python scripts/entrypoint.py trainer-attributes -- --config configs/trainers/attributes.yaml
```

## Design rules
- GPU is mandatory.
- Persistent intermediate caching is not supported.
- Ultralytics is the only supported detection, segmentation, and first local-tracker API.
- Documentation must live alongside implementation.
- Jupyter notebooks in `tests_notebooks/` are for component experiments and manual validation.

## Branching note
Foundation implementation work is being developed on `feat/v2-foundation`.
