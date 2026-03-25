# Neural Edge Inference

This document describes the first `rest_app` pipeline where the configurable reasoning module affects actual clustering decisions.

## What it does
The `neural_edge_inference` pipeline:
- runs Ultralytics detection on a real multi-camera frame-folder dataset
- builds the standard runtime spatial graph using handcrafted edge features
- converts that graph into a tensor batch internally
- runs the configured reasoning module for `direct_score`, `gnn`, or `hybrid`
- converts model logits into edge probabilities with a sigmoid
- overwrites graph edge scores with those probabilities
- clusters detections using the learned probabilities instead of the raw handcrafted weighted score

## Why this matters
This is the first step where the configurable neural layer changes actual tracking behavior rather than only serving as a diagnostic probe.

## Current limitations
- currently frame-graph only
- still inference-only
- no training loop yet
- no attribute or segmentation enrichment in tensorized features yet
- no checkpoint loading or trained model management yet
- the model is currently randomly initialized unless later integration supplies trained weights

## Practical implication
At this phase, `neural_edge_inference` is primarily an architectural and integration milestone, not yet a quality milestone. It proves that the configurable reasoning modules can participate in the real scoring path used for graph clustering.
