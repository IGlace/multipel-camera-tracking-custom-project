# Mcmt Platform Plan V2 - Implementation Expanded

## 1. Objective

Build a unified GPU-first multi-camera multi-target tracking platform with three separate apps and two trainer modules:

* `rest_app`: offline graph research app based on prepared annotations/detections
* `heuristic_app`: non-graph tracking app
* `realtime_app`: runtime detection/tracking app
* `trainers/reid_mgn`: ReID trainer/exporter
* `trainers/attributes`: attribute trainer/exporter

The platform must stay maintainable, strongly modular, and easy to extend without collapsing all logic into one giant pipeline.

---

## 2. Critical Architectural Correction

This is the most important correction to the previous plan.

### 2.1 `rest_app` does **not** run a detector or a local tracker

`rest_app` must follow the original ReST project logic more closely:

* it always receives prepared dataset annotations in MOT-style format
* those annotations include bounding boxes and track IDs
* graph construction is based on those prepared detections/IDs
* there is **no detection stage** inside `rest_app`
* there is **no local tracker stage** inside `rest_app`
* ID generation in `rest_app` is not produced by a detector/tracker pipeline like in `realtime_app`

So `rest_app` is an **offline graph-learning and graph-inference app over prepared structured data**, not a full online CV pipeline.

### 2.2 Where detection and tracking really belong

* `realtime_app`: detector + optional local tracker + online graph/global association
* `heuristic_app`: non-GNN/non-graph matching/tracking over real inputs, using direct logic rather than graph neural reasoning
* `rest_app`: graph learning/reasoning over prepared MOT-style annotations and crops

This separation must remain strict in the codebase.

---

## 3. Final App Boundaries

## 3.1 `rest_app`

### Role

Offline graph-native research app.

### Input contract

`rest_app` consumes prepared sequence data such as:

* image frames per camera
* MOT-style annotation files per camera
* bounding boxes per frame
* ground-truth track IDs
* optional calibration / homography / metadata files

### It must support

* graph dataset creation for training
* graph dataset creation for evaluation/testing
* offline graph inference over prepared detections
* frame-graph mode
* tracklet-graph mode
* direct-score, GNN, and hybrid edge reasoning
* optional ReID and attribute enrichment using crops extracted from annotated boxes

### It must not do

* object detection
* online tracking
* local tracker identity generation
* Ultralytics tracker usage for identity creation

### Internal modes

* `train`
* `eval`
* `test`
* `infer`

### Meaning of these modes

* `train`: build labeled graph samples from prepared annotations and optimize graph model
* `eval`: run validation metrics during training or model comparison
* `test`: run full offline evaluation on prepared labeled sequences
* `infer`: run graph reasoning on prepared detections without training step

### Submodules

* `apps/rest_app/frame_graph`
* `apps/rest_app/tracklet_graph`
* `apps/rest_app/pipelines`
* `apps/rest_app/runners`

---

## 3.2 `heuristic_app`

### Role

Non-graph, non-GNN tracking app.

### Input contract

Real inputs such as:

* multi-camera frame folders
* video files
* folders of videos
* optional prepared detections if later desired

### It must support

* frame-by-frame matching without graphs
* tracklet-based matching without graphs
* detector usage when needed
* direct scoring / thresholds / Hungarian / rule-based filtering
* output generation for real-world testing

### It must not do

* DGL usage
* graph neural message passing
* graph object reasoning as the main matching mechanism

### Internal modes

* `test`
* `infer`

### Submodules

* `apps/heuristic_app/frame_match`
* `apps/heuristic_app/tracklet_match`
* `apps/heuristic_app/runners`

---

## 3.3 `realtime_app`

### Role

Runtime-oriented online experimentation app.

### Input contract

* multi-camera frame folders
* single video file
* folder of videos

### It must support

* frame-graph runtime mode
* tracklet-graph runtime mode
* detector usage in frame mode
* local tracker usage in tracklet mode
* retroactive correction/reconciliation policies later
* live display and runtime outputs

### Internal modes

* `infer`
* `live`
* `batch_video`

### Submodules

* `apps/realtime_app/input_sources`
* `apps/realtime_app/frame_graph_runtime`
* `apps/realtime_app/tracklet_graph_runtime`
* `apps/realtime_app/local_tracking`
* `apps/realtime_app/global_association`
* `apps/realtime_app/reconciliation`
* `apps/realtime_app/runners`

---

## 4. Data Contracts

## 4.1 `rest_app` dataset contract

This should be explicit and stable.

A sequence should be normalized into something like:

```text
sequence_root/
  cameras/
    cam_01/
      000001.jpg
      000002.jpg
    cam_02/
      000001.jpg
      000002.jpg
  annotations/
    cam_01.txt
    cam_02.txt
  metadata/
    homography.json
    calibration.json
    sequence_info.yaml
```

Each annotation file should be normalized into a common record structure.

Example canonical row fields:

* `camera_id`
* `frame_index`
* `track_id_gt`
* `bbox_x`
* `bbox_y`
* `bbox_w`
* `bbox_h`
* `confidence` or `visibility` if available
* `class_id` if relevant

Even if the original source format changes, the adapter must convert it into one shared internal format.

### Important rule

Never confuse these two concepts:

* `track_id_gt`: identity from annotation/ground truth
* `pred_track_id`: identity predicted by the model or inference logic

These must never share the same field name internally.

## 4.2 `heuristic_app` and `realtime_app` dataset contract

These apps should normalize raw inputs into a shared runtime batch object such as:

* `camera_id`
* `frame_index`
* `timestamp`
* `image`
* `image_path` or `video_source`
* optional prepared detections
* optional optional GT only for evaluation, not for core operation

---

## 5. Graph Strategy by App

## 5.1 `rest_app` frame graph

### Nodes

Each node should represent one annotated detection from the prepared dataset.

### Edges

Spatial edges:

* between candidate detections at the same timestamp across cameras

Temporal edges:

* between candidate detections or track states across time windows

### Labels

Labels should come from GT IDs.

Examples:

* positive spatial edge: same GT identity across two cameras at same synchronized time
* negative spatial edge: different GT identity
* positive temporal edge: same GT identity across time
* negative temporal edge: different GT identity

### Implementation note

Do not let model code compute labels.

Label creation must happen in dataset/preprocessing code, before model forward.

## 5.2 `rest_app` tracklet graph

### Nodes

Each node should represent a prepared offline tracklet object built from annotated data.

Tracklets can be built from GT-linked detections over windows, or from prepared offline association files if you later support them.

### Tracklet fields should include

* `camera_id`
* `track_id_gt`
* `start_frame`
* `end_frame`
* `length`
* representative bbox statistics
* motion statistics
* optional crop-level pooled ReID/attribute features

### Important implementation note

The tracklet builder for `rest_app` is not the same as the online tracklet builder for `realtime_app`.

* `rest_app` tracklets are **offline dataset objects**
* `realtime_app` tracklets are **online rolling runtime objects**

Keep them in separate submodules even if they share some helper utilities.

## 5.3 `heuristic_app`

No graph object is required.

The conceptual stages may mirror graph logic, but the actual implementation should remain:

* list-based
* tensor-based
* matrix-based
* assignment-based

## 5.4 `realtime_app`

Frame mode:

* detector -> features -> graph/global association

Tracklet mode:

* detector + local tracker -> rolling tracklets -> global association -> reconciliation

---

## 6. Feature Engineering Strategy

## 6.1 Node features

Node feature composition must be configurable.

Possible node components:

* bbox center
* bbox area
* bbox width / height
* aspect ratio
* confidence
* class id
* frame embedding
* camera embedding
* homography projection
* velocity
* acceleration
* tracklet duration
* ReID embedding
* attribute vector
* segmentation descriptors

### Important code tip

Do not make one huge node-builder function with 30 `if` statements.

Use a component registry pattern.

Example shape:

```python
NODE_COMPONENTS = {
    "center_x": build_center_x,
    "center_y": build_center_y,
    "area": build_area,
    "reid": build_reid_embedding,
    "attributes": build_attribute_vector,
}
```

Then compose features from a selected ordered list.

This keeps tensor dimensionality deterministic.

## 6.2 Edge features

Edge features must also be component-based.

Possible edge components:

* IoU
* center distance / similarity
* width ratio similarity
* height ratio similarity
* area ratio similarity
* appearance cosine similarity
* appearance L2 distance
* attribute similarity
* attribute agreement ratio
* temporal gap
* velocity difference
* homography distance
* camera-pair encoding

### Important code tip

Return both:

* a `dict[str, float]` for debugging/readability
* a tensorized ordered vector for the model

That way you keep explainability for notebook/debug sinks without losing model compatibility.

---

## 7. ReID and Attribute Integration

## 7.1 General rule

ReID and attributes are not separate decoration modules. They must plug into the feature path cleanly.

## 7.2 In `rest_app`

Because `rest_app` already has boxes from annotations, the usual pattern is:

* crop the bbox from the image
* feed crop to ReID extractor if enabled
* feed crop to attribute extractor if enabled
* store outputs in node metadata and optionally in node feature vectors
* derive edge features from them when enabled

## 7.3 In `realtime_app`

Same idea, but crops come from detector or tracker outputs.

## 7.4 In `heuristic_app`

Same idea, but no graph layer is involved. ReID and attributes directly influence matching costs.

### Important code tip

Use lazy optional builders.

Example:

```python
reid_extractor = maybe_build_reid_extractor(cfg.reid)
attr_extractor = maybe_build_attribute_extractor(cfg.attributes)
```

Then downstream code must work when they are `None`.

---

## 8. GNN Strategy for `rest_app`

## 8.1 Supported reasoning modes

* `direct_score`
* `gnn`
* `hybrid`

## 8.2 Practical meaning

### `direct_score`

Use tensorized edge features and score them with a direct MLP or even weighted rules.

### `gnn`

Use node + edge tensors, message passing, updated edge states, predictor head.

### `hybrid`

Combine direct score branch and graph branch.

---

## 8.3 Recommended implementation split

Do not mix model construction with tracking pipeline construction.

Keep these layers separate:

1. dataset/sample builder
2. graph object builder
3. graph-to-tensor converter
4. model factory
5. inference/training runner
6. clustering / association postprocessor

### Good separation example

```python
sample = dataset[idx]
graph = graph_builder(sample)
tensor_batch = graph_to_tensor(graph)
outputs = model(tensor_batch)
association = postprocessor(graph, outputs)
```

This is much easier to debug than hiding everything inside one `run()` function.

---

## 8.4 GNN code tips

### Tip 1: keep block configs uniform

Use the same config structure for:

* node encoder
* edge encoder
* message encoder
* node updater
* edge updater
* predictor

### Tip 2: make tensor shapes explicit

Inside model code, comment expected shapes.

Example:

```python
# node_features: [N, Dn]
# edge_index: [2, E]
# edge_features: [E, De]
```

### Tip 3: keep aggregation isolated

Do not hide aggregation logic inside a large forward pass.

Create one helper for:

* sum
* mean
* max

### Tip 4: make hybrid fusion explicit

Do not bury hybrid fusion inside unrelated code.

Example:

```python
fused_logits = alpha * gnn_logits + (1 - alpha) * direct_logits
```

### Tip 5: keep edge ordering stable

The edge order used in:

* graph object
* tensor conversion
* output logits must remain identical.

If edge ordering drifts, debugging becomes painful.

---

## 9. Output and Visualization Strategy

Sinks should stay sink-based and independent.

Supported sinks:

* MOT text sink
* annotated frame sink
* annotated video sink
* live grid sink
* graph debug sink
* evaluation report sink
* run manifest sink

### Important code tip

Let every sink accept a generic optional payload.

Example:

```python
payload = {
    "graph": graph,
    "tensor_batch": tensor_batch,
    "model_outputs": outputs,
}
```

Graph-specific sinks can use it. Other sinks can ignore it.

---

## 10. Trainers Integration Strategy

## 10.1 Standardized checkpoint package

All trainer exports should use one package contract.

Example fields:

* `state_dict`
* `metadata`
* `model_config`
* optional `class_names`

### Why this matters

This lets runtime code load models without depending on the original training repo layout.

## 10.2 `trainers/reid_mgn`

Responsibilities:

* training logic
* evaluation logic
* export standardized package
* record enough metadata for runtime extractor reconstruction

## 10.3 `trainers/attributes`

Same principle.

### Important code tip

Export first, full training integration second.

This reduces coupling and lets runtime code stabilize earlier.

---

## 11. Detailed Implementation Notes Per App

## 11.1 `rest_app` recommended execution order

### Frame graph training/testing

1. parse sequence metadata
2. parse MOT annotation files
3. group detections by synchronized timestamp
4. optionally crop boxes for ReID/attributes
5. build node records
6. build spatial and/or temporal edge candidates
7. generate GT edge labels
8. tensorize graph
9. run model
10. compute loss / metrics
11. postprocess / save outputs

### Frame graph inference

1. parse prepared detections from annotation-like files
2. build nodes and candidate edges
3. tensorize graph
4. score edges with selected mode
5. threshold / cluster / associate
6. save MOT / graph debug / reports

### Tracklet graph

Same idea, but graph nodes are tracklets, not raw detections.

## 11.2 `heuristic_app` recommended execution order

### Frame match

1. load images
2. detect objects if needed
3. optional crop-based ReID/attributes
4. build cost matrix / rule-based scores
5. assign matches
6. update identity state
7. save outputs

### Tracklet match

1. detect/tracker or prepared detections
2. build local short tracklets
3. compute inter-tracklet costs
4. associate
5. save outputs

## 11.3 `realtime_app` recommended execution order

### Frame graph runtime

1. read runtime input frame(s)
2. detect objects
3. optional enrichments
4. build graph
5. score edges
6. cluster/associate
7. emit outputs

### Tracklet graph runtime

1. read frame(s)
2. run local tracker per camera
3. update rolling tracklets
4. enrich active tracklets if needed
5. build tracklet graph
6. global association
7. reconciliation logic later
8. emit outputs

---

## 12. Suggested Core Data Classes

These should exist early and remain stable.

### Dataset/data records

* `AnnotatedDetectionRecord`
* `RuntimeDetectionRecord`
* `LocalTrackObservation`
* `TrackletRecord`
* `SequenceSample`
* `MultiCameraFrameBatch`

### Graph records

* `NodeRecord`
* `EdgeRecord`
* `GraphSample`
* `GraphTensorBatch`

### Output records

* `TrackObservation`
* `EvaluationSummary`
* `RunManifest`

### Important code tip

Use dataclasses for records, not giant mutable dictionaries everywhere.

Dictionaries are fine for metadata payloads, but not for your primary contracts.

---

## 13. Suggested Directory Layout

```text
project_root/
  apps/
    rest_app/
      __init__.py
      main.py
      frame_graph/
        pipeline.py
        neural_pipeline.py
        dataset.py
        labels.py
      tracklet_graph/
        pipeline.py
        dataset.py
        labels.py
      pipelines/
      runners/
    heuristic_app/
      __init__.py
      main.py
      frame_match/
        pipeline.py
      tracklet_match/
        pipeline.py
      runners/
    realtime_app/
      __init__.py
      main.py
      input_sources/
      frame_graph_runtime/
        pipeline.py
      tracklet_graph_runtime/
        pipeline.py
      local_tracking/
      global_association/
      reconciliation/
      runners/
  mcmt_core/
    config/
    logging/
    io/
    datasets/
    detections/
    segmentation/
    reid/
    attributes/
    features/
    graphs/
    matching/
    tracklets/
    visualization/
    outputs/
    evaluation/
    runtime/
    utils/
  trainers/
    reid_mgn/
    attributes/
  configs/
    rest/
    heuristic/
    realtime/
    trainers/
  scripts/
  tests_notebooks/
  outputs/
  Dockerfile
  docker-compose.yml
```

---

## 14. Configuration Strategy

### 14.1 Important domains

* `system`
* `dataset`
* `detector`
* `segmentor`
* `local_tracker`
* `reid`
* `attributes`
* `tracklet`
* `graph_model`
* `outputs`
* `logging`
* `evaluation`

### 14.2 Validation rules that matter a lot

Reject these combinations early:

* `heuristic_app` with GNN reasoning mode
* `heuristic_app` with graph debug sink
* `rest_app` trying to use detector or local tracker as required core stage
* tracklet-only node features in pure frame mode
* attribute edge features enabled without attribute backend
* ReID edge features enabled without ReID backend
* realtime tracklet mode without local tracker config
* GNN mode without graph tensor conversion path

### Important code tip

Do not wait until runtime crash to catch invalid config.

Validate on config load.

---

## 15. Updated Implementation Phases

### Phase 1 - Foundation

* project skeleton
* config loader + validator
* logging
* GPU check
* CLI structure
* Docker
* notebook skeleton

### Phase 2 - Core backends

* Ultralytics wrappers
* dataset adapters
* output sink interfaces
* ReID runtime extractor
* attribute runtime extractor
* standardized checkpoint packaging

### Phase 3 - Offline graph core for `rest_app`

* parse MOT-style annotations
* build annotated detection records
* build frame graph samples
* build tracklet graph samples
* GT edge labeling
* graph tensor conversion

### Phase 4 - Direct-score baselines

* `rest_app` direct-score frame graph baseline
* `heuristic_app` frame and tracklet baselines
* `realtime_app` frame and tracklet baselines

### Phase 5 - Neural graph layer

* configurable MLP blocks
* direct scorer module
* GNN module
* hybrid module
* model factory
* graph probe pipeline
* neural edge inference experiment path

### Phase 6 - Enrichment layer

* ReID crop extraction integration
* attribute crop extraction integration
* node enrichment
* edge enrichment
* enriched notebook experiments

### Phase 7 - Trainer parity and export

* ReID trainer parity
* attribute trainer parity
* export package validation
* runtime load validation

### Phase 8 - Parity and cleanup

* compare behavior with original ReST where relevant
* tighten directory responsibilities
* remove duplicated logic
* improve docs and examples

---

## 16. Practical Code Tips

### Tip A

Write small pure helpers first, then orchestration later.

### Tip B

Keep I/O separate from math.

Parsing files, building graphs, scoring edges, and saving outputs should not all happen in the same function.

### Tip C

Prefer stable intermediate contracts.

Bad:

* one function returns random dictionaries with changing keys

Good:

* one function returns `NodeRecord`
* one returns `EdgeRecord`
* one returns `GraphTensorBatch`

### Tip D

Never hide GT semantics.

Use explicit field names:

* `track_id_gt`
* `pred_track_id`
* `edge_label_gt`

### Tip E

Use ordered feature lists for tensorization.

Never rely on raw dictionary iteration order for model inputs.

### Tip F

Keep one source of truth for dimensions.

If node feature components change, node input dimension should be derived automatically from the selected ordered component list.

### Tip G

Do not overgeneralize too early.

It is fine to build:

* one clean `rest_app` frame graph path first
* then extend to tracklet graph

### Tip H

Keep debug visibility high.

During development, save:

* graph statistics
* a few edge score previews
* tensor shapes
* a few node/edge examples

### Tip I

Do not mix runtime and offline assumptions.

If logic depends on GT IDs, it belongs to `rest_app` or offline evaluation, not to online runtime tracking.

---

## 17. Final Non-Negotiable Rules

* One Dockerfile only.
* GPU required.
* No intermediate caching.
* `rest_app` is annotation-driven, not detector/tracker-driven.
* Ultralytics only for detector, segmentor, and first local tracker implementation.
* `rest_app`, `heuristic_app`, and `realtime_app` remain separate apps.
* Shared logic goes into `mcmt_core` only.
* Sinks remain sink-based and multi-enabled.
* Config validation must reject incompatible combinations early.
* GT IDs and predicted IDs must never be conflated.

---

## 18. Reference projects

- Rest Project : https://github.com/chengche6230/ReST
- MGN Project : https://github.com/GNAYUOHZ/ReID-MGN
- Attribute Recognition Project : https://github.com/hiennguyen9874/person-attribute-recognition