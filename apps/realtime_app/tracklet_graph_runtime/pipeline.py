"""First usable tracklet-runtime baseline for realtime_app.

This phase implements a rolling tracklet runtime using one Ultralytics local tracker per camera.
The tracker outputs are converted into rolling tracklet summaries, then cross-camera graph
association is run on the active tracklet summaries.
"""

from __future__ import annotations

from mcmt_core.config.schema import RuntimeConfig
from mcmt_core.datasets import MultiCameraFrameDataset, MultiCameraVideoDataset
from mcmt_core.features import EdgeFeatureBuilder
from mcmt_core.graphs import build_spatial_frame_graph
from mcmt_core.matching import ClusterState, LocalTrackObservation, UltralyticsLocalTracker, match_clusters
from mcmt_core.outputs import TrackObservation
from mcmt_core.runtime import build_output_manager, cluster_nodes_graph, to_observations
from mcmt_core.tracklets import SlidingWindowTrackletBuilder, TrackletRecord


def _build_dataset(cfg: RuntimeConfig):
    if cfg.dataset.input_type == "frame_folders":
        return MultiCameraFrameDataset(cfg.dataset)
    if cfg.dataset.input_type in {"video_file", "video_folder"}:
        return MultiCameraVideoDataset(cfg.dataset)
    raise ValueError(f"Unsupported realtime dataset input_type: {cfg.dataset.input_type}")


def _build_local_trackers(cfg: RuntimeConfig, camera_ids: list[str]) -> dict[str, UltralyticsLocalTracker]:
    return {
        camera_id: UltralyticsLocalTracker(
            model=cfg.detector.model,
            tracker=cfg.local_tracker.tracker,
            tracker_config=cfg.local_tracker.tracker_config,
            conf=cfg.detector.confidence,
            iou=cfg.detector.iou,
        )
        for camera_id in camera_ids
    }


def _tracklet_observations(cluster_states: list[ClusterState], active_tracklets: list[TrackletRecord]) -> dict[str, list[TrackObservation]]:
    latest_by_key = {
        (tracklet.camera_id, tracklet.local_track_id): tracklet.observations[-1]
        for tracklet in active_tracklets
        if tracklet.observations
    }
    observations: dict[str, list[TrackObservation]] = {}
    for state in cluster_states:
        for tracklet in state.nodes:
            latest = latest_by_key.get((tracklet.camera_id, tracklet.local_track_id))
            if latest is None:
                continue
            observations.setdefault(tracklet.camera_id, []).append(
                TrackObservation(
                    camera_id=tracklet.camera_id,
                    frame_index=latest.frame_index,
                    timestamp=latest.timestamp,
                    track_id=state.global_id,
                    bbox_xyxy=latest.bbox_xyxy,
                    confidence=latest.confidence,
                    class_id=latest.class_id,
                    class_name=latest.class_name,
                )
            )
    return observations


def run_realtime_tracklet_runtime_baseline(cfg: RuntimeConfig, logger, mode: str = "infer") -> None:
    dataset = _build_dataset(cfg)
    iterator = iter(dataset)
    try:
        first_batch = next(iterator)
    except StopIteration:
        logger.warning("The configured realtime dataset is empty.")
        return

    camera_ids = [frame.camera_id for frame in first_batch.frames]
    trackers = _build_local_trackers(cfg, camera_ids)
    builder = SlidingWindowTrackletBuilder(
        window_size=cfg.tracklet.window_size,
        min_length=cfg.tracklet.min_length,
        idle_tolerance=cfg.tracklet.idle_tolerance,
    )
    edge_builder = EdgeFeatureBuilder(
        selected_features=cfg.graph_model.temporal_edge_features,
        score_weights=cfg.graph_model.score_weights,
    )
    if mode == "live":
        cfg.outputs.enable_live_display = True
    output_manager = build_output_manager(cfg)

    previous_clusters: list[ClusterState] = []
    next_global_id = 1

    def process_batch(batch):
        nonlocal previous_clusters, next_global_id
        images_by_camera = {frame.camera_id: frame.image for frame in batch.frames}
        local_observations: list[LocalTrackObservation] = []
        for frame in batch.frames:
            tracker = trackers[frame.camera_id]
            local_observations.extend(
                tracker.track(
                    frame.image,
                    camera_id=frame.camera_id,
                    frame_index=batch.frame_index,
                    timestamp=batch.timestamp,
                )
            )
        active_tracklets = builder.update(batch.frame_index, local_observations)
        graph = build_spatial_frame_graph(active_tracklets, edge_builder)
        current_clusters = cluster_nodes_graph(graph, cfg.graph_model.edge_score_threshold)
        current_states, next_global_id = match_clusters(
            previous_clusters,
            current_clusters,
            cfg.graph_model.temporal_match_threshold,
            next_global_id,
        )
        observations_by_camera = _tracklet_observations(current_states, active_tracklets)
        output_manager.write(
            timestamp=batch.timestamp,
            frame_index=batch.frame_index,
            images_by_camera=images_by_camera,
            observations_by_camera=observations_by_camera,
        )
        previous_clusters = current_states
        logger.info(
            "tracklet_frame=%s local_obs=%d active_tracklets=%d edges=%d clusters=%d next_global_id=%d",
            batch.timestamp,
            len(local_observations),
            len(active_tracklets),
            graph.number_of_edges(),
            len(current_states),
            next_global_id,
        )

    logger.info(
        "Running tracklet_runtime_baseline with dataset.input_type=%s",
        cfg.dataset.input_type,
    )

    try:
        process_batch(first_batch)
        for batch in iterator:
            process_batch(batch)
    finally:
        output_manager.close()
