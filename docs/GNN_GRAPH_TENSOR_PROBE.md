# Graph Tensor Probe

This document describes the first bridge between the runtime tracking graphs and the configurable neural reasoning scaffold.

## What it does
The `graph_tensor_probe` pipeline in `rest_app`:
- runs Ultralytics detection on a real multi-camera frame-folder dataset
- builds the normal runtime spatial graph
- converts that graph into a `GraphTensorBatch`
- builds the requested reasoning module for `direct_score`, `gnn`, or `hybrid`
- runs a forward pass and logs output tensor shapes and a small logits preview

## Why this matters
This is the first point where the configurable neural stack touches real graph data rather than synthetic dummy tensors.

It does not yet perform training, but it stabilizes:
- node feature tensorization
- edge feature tensorization
- graph-to-tensor conversion
- per-mode forward signatures

## Current limitations
- currently frame-graph only
- currently probe-oriented, not a training loop
- currently limited to a small number of processed timestamps
- not yet integrated into the main tracking decision path
