# GNN Scaffold for rest_app

This document describes the first configurable neural-model scaffold added for `rest_app`.

## Supported reasoning modes
- `direct_score`
- `gnn`
- `hybrid`

## What is implemented now
The current scaffold provides:
- configurable MLP blocks
- configurable direct edge scorer
- configurable message-passing graph network
- configurable hybrid fusion model
- model factory that selects the correct module for the requested reasoning mode
- a runnable `gnn_sanity` pipeline that builds the model and runs a dummy forward pass

## What is not integrated yet
This scaffold does **not** yet replace the current frame-graph baseline.

The current frame-graph baseline still uses direct feature scoring from the heuristic edge builder.
The neural scaffold is currently intended for:
- architecture stabilization
- config stabilization
- notebook experiments
- later training and inference integration

## Configurable block families
The config currently supports these neural block groups:
- direct scorer
- node encoder
- edge encoder
- message encoder
- node updater
- edge updater
- predictor

Each block supports:
- `hidden_dims`
- `output_dim`
- `activation`
- `norm`
- `dropout`
- `residual`
- `activate_last`

## Next integration step
The next major step after this scaffold is to connect the neural reasoning modules to actual graph-tensor batches derived from the tracking graphs, while preserving the current direct-score baseline as a fallback reference.
