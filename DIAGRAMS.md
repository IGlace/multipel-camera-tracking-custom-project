# MCMT App Separation Documentation

This document separates the responsibilities of each app in the platform and shows, with Mermaid diagrams, what each app can do, what it must not do, and how execution changes depending on configuration.

---

## 1. High-level separation

```mermaid
flowchart TB
    A[MCMT Platform] --> B[rest_app]
    A --> C[heuristic_app]
    A --> D[realtime_app]
    A --> E[trainers/reid_mgn]
    A --> F[trainers/attributes]

    B --> B1[Offline graph learning/inference]
    B --> B2[Prepared MOT-style annotations]
    B --> B3[Frame graph or tracklet graph]

    C --> C1[Non-graph tracking]
    C --> C2[Direct matching and assignment]
    C --> C3[Frame mode or tracklet mode]

    D --> D1[Online/runtime processing]
    D --> D2[Detection and optional local tracking]
    D --> D3[Frame graph or tracklet graph runtime]

    E --> E1[Train/export ReID backend]
    F --> F1[Train/export attribute backend]
```

---

## 2. Responsibility matrix

| Component             | Main role                      | Input style                                 | Uses detector  | Uses local tracker   | Uses graphs | Uses GNN       | Uses GT IDs directly     |
| --------------------- | ------------------------------ | ------------------------------------------- | -------------- | -------------------- | ----------- | -------------- | ------------------------ |
| `rest_app`            | Offline graph research         | Prepared MOT-style annotations              | No             | No                   | Yes         | Optional       | Yes                      |
| `heuristic_app`       | Non-graph tracking             | Real inputs or optional prepared detections | Yes, if needed | Optional later       | No          | No             | No                       |
| `realtime_app`        | Runtime online experimentation | Frames/videos                               | Yes            | Yes in tracklet mode | Yes         | Optional later | No                       |
| `trainers/reid_mgn`   | ReID training/export           | Labeled ReID training data                  | No             | No                   | No          | No             | Depends on training data |
| `trainers/attributes` | Attribute training/export      | Labeled attribute training data             | No             | No                   | No          | No             | Depends on training data |

---

## 3. `rest_app`

## 3.1 What `rest_app` is for

```mermaid
flowchart LR
    A[rest_app] --> B[Offline graph dataset creation]
    A --> C[Offline graph training]
    A --> D[Offline graph evaluation]
    A --> E[Offline graph inference]
    A --> F[Frame graph mode]
    A --> G[Tracklet graph mode]
    A --> H[Direct-score mode]
    A --> I[GNN mode]
    A --> J[Hybrid mode]
    A --> K[Optional ReID enrichment]
    A --> L[Optional attribute enrichment]
```

## 3.2 What `rest_app` must never do

```mermaid
flowchart TD
    A[rest_app forbidden tasks] --> B[Do not run object detector]
    A --> C[Do not run online local tracker]
    A --> D[Do not generate identities from detector tracker flow]
    A --> E[Do not behave like realtime pipeline]
    A --> F[Do not depend on runtime-only assumptions]
```

## 3.3 `rest_app` input contract

```mermaid
flowchart TD
    A[Prepared sequence dataset] --> B[Camera frames]
    A --> C[MOT-style annotation files]
    A --> D[Ground-truth track IDs]
    A --> E[Optional metadata]

    E --> E1[Homography]
    E --> E2[Calibration]
    E --> E3[Sequence info]

    C --> F[Canonical annotated detection records]
    F --> G[Graph sample builder]
```

## 3.4 `rest_app` config-driven process

```mermaid
flowchart TD
    A[Start rest_app] --> B[Load config]
    B --> C{mode}

    C -->|train| D[Build labeled graph samples]
    C -->|eval| E[Build validation graph samples]
    C -->|test| F[Build labeled test graph samples]
    C -->|infer| G[Build unlabeled or inference graph samples]

    D --> H{graph type}
    E --> H
    F --> H
    G --> H

    H -->|frame_graph| I[Build detection nodes from annotations]
    H -->|tracklet_graph| J[Build offline tracklet nodes from annotations]

    I --> K{enrichment}
    J --> K

    K -->|none| L[Use geometric and metadata features only]
    K -->|reid enabled| M[Crop boxes and extract ReID]
    K -->|attributes enabled| N[Crop boxes and extract attributes]
    K -->|both enabled| O[Extract ReID and attributes]

    L --> P{reasoning mode}
    M --> P
    N --> P
    O --> P

    P -->|direct_score| Q[Score edges directly]
    P -->|gnn| R[Tensorize graph and run GNN]
    P -->|hybrid| S[Run direct scorer and GNN then fuse]

    Q --> T[Threshold cluster associate]
    R --> T
    S --> T

    T --> U[Evaluation outputs or inference outputs]
```

## 3.5 `rest_app` frame graph process

```mermaid
flowchart TD
    A[Annotated detections grouped by timestamp] --> B[Build node records]
    B --> C[Generate spatial edge candidates]
    C --> D[Generate temporal edge candidates if enabled]
    D --> E[Generate GT edge labels]
    E --> F[Optional crop-based enrichment]
    F --> G[Build graph object]
    G --> H[Convert to tensor batch]
    H --> I[Run reasoning module]
    I --> J[Score edges]
    J --> K[Cluster or associate]
    K --> L[Metrics and outputs]
```

## 3.6 `rest_app` tracklet graph process

```mermaid
flowchart TD
    A[Annotated detections with GT IDs] --> B[Offline tracklet builder]
    B --> C[Tracklet node records]
    C --> D[Tracklet edge candidates]
    D --> E[Tracklet GT labels]
    E --> F[Optional pooled ReID and attributes]
    F --> G[Tracklet graph]
    G --> H[Tensorization]
    H --> I[Reasoning module]
    I --> J[Association or clustering]
    J --> K[Metrics and outputs]
```

## 3.7 `rest_app` capability summary

* Builds offline graph datasets from prepared annotations
* Can train, evaluate, test, and infer
* Supports frame graph and tracklet graph
* Supports direct score, GNN, and hybrid reasoning
* Can enrich nodes and edges with ReID and attributes
* Uses GT IDs for labeling and evaluation
* Never runs detector or runtime tracker

---

## 4. `heuristic_app`

## 4.1 What `heuristic_app` is for

```mermaid
flowchart LR
    A[heuristic_app] --> B[Non-graph tracking]
    A --> C[Frame-by-frame matching]
    A --> D[Tracklet-based matching]
    A --> E[Rule-based scoring]
    A --> F[Cost matrices]
    A --> G[Assignment methods]
    A --> H[Optional detector use]
    A --> I[Optional ReID and attribute enrichment]
```

## 4.2 What `heuristic_app` must never do

```mermaid
flowchart TD
    A[heuristic_app forbidden tasks] --> B[Do not use graph objects as main association structure]
    A --> C[Do not use graph neural message passing]
    A --> D[Do not use DGL as core dependency]
    A --> E[Do not behave like offline labeled graph trainer]
```

## 4.3 `heuristic_app` config-driven process

```mermaid
flowchart TD
    A[Start heuristic_app] --> B[Load config]
    B --> C{input type}

    C -->|frame folders| D[Load multi-camera frames]
    C -->|video file| E[Read video frames]
    C -->|video folder| F[Read one or more videos]
    C -->|prepared detections later optional| G[Normalize external detections]

    D --> H{matching mode}
    E --> H
    F --> H
    G --> H

    H -->|frame_match| I[Build per-frame candidate detections]
    H -->|tracklet_match| J[Build local short tracklets]

    I --> K{detector available}
    J --> L{source of local tracklets}

    K -->|yes| M[Run detector]
    K -->|no prepared detections| N[Use prepared detections]

    L -->|detector plus internal logic| O[Create short tracklets]
    L -->|prepared detections later optional| P[Create tracklets from supplied detections]

    M --> Q{enrichment}
    N --> Q
    O --> Q
    P --> Q

    Q -->|none| R[Use geometric and score rules]
    Q -->|reid| S[Add appearance costs]
    Q -->|attributes| T[Add attribute costs]
    Q -->|both| U[Add appearance and attribute costs]

    R --> V[Build cost matrix or rules]
    S --> V
    T --> V
    U --> V

    V --> W[Assignment thresholding and identity update]
    W --> X[Outputs]
```

## 4.4 `heuristic_app` frame match process

```mermaid
flowchart TD
    A[Input frames] --> B[Detector or prepared detections]
    B --> C[Optional crop enrichment]
    C --> D[Build pairwise costs]
    D --> E[Apply thresholds rules Hungarian or equivalent]
    E --> F[Update identities]
    F --> G[Save MOT and visual outputs]
```

## 4.5 `heuristic_app` tracklet match process

```mermaid
flowchart TD
    A[Input frames or detections] --> B[Create local short tracklets]
    B --> C[Compute inter-tracklet costs]
    C --> D[Apply matching rules]
    D --> E[Update global identities]
    E --> F[Save outputs]
```

## 4.6 `heuristic_app` capability summary

* Pure non-graph tracking app
* Good for baseline comparisons
* Can use detector if needed
* Can use ReID and attributes directly in costs
* Supports frame mode and tracklet mode
* Does not do GNN or graph reasoning

---

## 5. `realtime_app`

## 5.1 What `realtime_app` is for

```mermaid
flowchart LR
    A[realtime_app] --> B[Runtime online experimentation]
    A --> C[Frame graph runtime mode]
    A --> D[Tracklet graph runtime mode]
    A --> E[Detector-driven flow]
    A --> F[Local tracker-driven flow in tracklet mode]
    A --> G[Global association]
    A --> H[Optional reconciliation later]
    A --> I[Live display]
    A --> J[MOT and visual outputs]
```

## 5.2 `realtime_app` frame-mode process

```mermaid
flowchart TD
    A[Frames or video input] --> B[Run detector]
    B --> C[Optional crop enrichment]
    C --> D[Build runtime nodes]
    D --> E[Build runtime graph]
    E --> F{reasoning mode}
    F -->|direct_score| G[Direct edge scoring]
    F -->|gnn later| H[Tensorize and run model]
    F -->|hybrid later| I[Fuse direct and neural scores]
    G --> J[Associate cluster]
    H --> J
    I --> J
    J --> K[Emit runtime outputs]
```

## 5.3 `realtime_app` tracklet-mode process

```mermaid
flowchart TD
    A[Frames or video input] --> B[Run detector plus local tracker per camera]
    B --> C[Update rolling local tracklets]
    C --> D[Optional enrichment on active tracklets]
    D --> E[Build cross-camera tracklet graph]
    E --> F[Global association]
    F --> G[Optional reconciliation correction later]
    G --> H[Emit outputs and live display]
```

## 5.4 `realtime_app` config-driven process

```mermaid
flowchart TD
    A[Start realtime_app] --> B[Load config]
    B --> C{input type}

    C -->|frame folders| D[Read synchronized frame folders]
    C -->|video file| E[Read one video]
    C -->|video folder| F[Read one or many videos]

    D --> G{runtime mode}
    E --> G
    F --> G

    G -->|frame_runtime| H[Detector-first frame pipeline]
    G -->|tracklet_runtime| I[Detector plus local tracker pipeline]

    H --> J{enrichment}
    I --> J

    J -->|none| K[Geometry only]
    J -->|reid| L[Appearance enrichment]
    J -->|attributes| M[Attribute enrichment]
    J -->|both| N[Appearance plus attributes]

    K --> O[Association]
    L --> O
    M --> O
    N --> O

    O --> P{outputs}
    P -->|mot| Q[MOT sink]
    P -->|frames| R[Annotated frame sink]
    P -->|video| S[Annotated video sink]
    P -->|live| T[Live grid sink]
    P -->|graph debug| U[Graph debug sink]
```

## 5.5 `realtime_app` capability summary

* Handles real online or batch runtime inputs
* Runs detector in frame mode
* Runs local tracker in tracklet mode
* Builds graph association online
* Supports multiple visual and runtime sinks
* Best place for future correction/reconciliation policies

---

## 6. Trainers

## 6.1 Trainer separation

```mermaid
flowchart LR
    A[Trainer modules] --> B[trainers/reid_mgn]
    A --> C[trainers/attributes]

    B --> B1[Train ReID backend]
    B --> B2[Evaluate ReID backend]
    B --> B3[Export standardized checkpoint package]

    C --> C1[Train attribute backend]
    C --> C2[Evaluate attribute backend]
    C --> C3[Export standardized checkpoint package]
```

## 6.2 Trainer to runtime relationship

```mermaid
flowchart TD
    A[Trainer output checkpoint] --> B[Standardized checkpoint package]
    B --> C[Runtime ReID extractor]
    B --> D[Runtime attribute extractor]
    C --> E[Node enrichment]
    D --> E
    E --> F[Edge enrichment]
    F --> G[Graph scoring or matching costs]
```

---

## 7. Config-based separation summary

```mermaid
flowchart TD
    A[Chosen app] --> B{app name}

    B -->|rest_app| C[Prepared annotations only]
    B -->|heuristic_app| D[Non-graph matching]
    B -->|realtime_app| E[Runtime detector tracker association]

    C --> C1{graph type}
    C1 -->|frame_graph| C2[Detection nodes from MOT annotations]
    C1 -->|tracklet_graph| C3[Offline tracklet nodes from annotations]

    D --> D1{matching type}
    D1 -->|frame_match| D2[Frame assignment logic]
    D1 -->|tracklet_match| D3[Tracklet assignment logic]

    E --> E1{runtime type}
    E1 -->|frame_runtime| E2[Detector then runtime graph]
    E1 -->|tracklet_runtime| E3[Local tracker then rolling tracklets]
```

---

## 8. Practical implementation notes from this separation

### 8.1 `rest_app`

* Create dedicated offline dataset adapters for MOT annotation parsing
* Keep GT label creation inside dataset/preprocessing code
* Keep detector/tracker imports out of the `rest_app` core path
* Add offline crop extraction utilities for ReID and attributes

### 8.2 `heuristic_app`

* Keep matching logic matrix-based or rule-based
* Do not reuse graph builders as the main internal representation
* ReID and attributes should directly modify costs or assignment filters

### 8.3 `realtime_app`

* Keep runtime input readers separate from graph logic
* Keep local tracker state separate from global association state
* Reconciliation should be a separate module, not mixed into local tracking

### 8.4 trainers

* Export standardized packages early
* Let runtime adapters depend on package format, not trainer code layout

---

## 9. Final non-negotiable separation rules

* `rest_app` is annotation-driven and GT-aware
* `rest_app` does not run detector or local tracker
* `heuristic_app` is non-graph and non-GNN
* `realtime_app` is runtime CV and association logic
* trainers only train/export backends
* shared utilities belong in `mcmt_core`
* app-specific orchestration must stay inside the corresponding app
