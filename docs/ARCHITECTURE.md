# V2 Architecture Notes

This repository follows the V2 plan for the consolidated MCMT platform.

## Main packages
- `mcmt_core`: shared reusable infrastructure and modeling primitives
- `apps/rest_app`: graph-native ReST-style tracking
- `apps/heuristic_app`: non-graph matching and tracking
- `apps/realtime_app`: runtime-oriented online tracking over frames and videos
- `trainers/reid_mgn`: dedicated MGN trainer/backend
- `trainers/attributes`: dedicated attribute trainer/backend

## Foundation scope in this branch
The current branch establishes:
- packaging
- config loading
- logging setup
- GPU bootstrap
- app and trainer entrypoints
- run manifest writing
- base config files
- notebook layer bootstrap

Core modeling modules, Ultralytics wrappers, and graph logic arrive in later commits.
