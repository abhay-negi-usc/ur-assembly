"""Perception layer: camera capture, ArUco, SAM3 cable/connector detection, multi-view fusion.

Imports are LAZY so the pure-numpy parts (ConnectorEstimator, the fusion math) can be used --
and tested -- without opencv or pyrealsense2 installed. `import urlab.perception` costs nothing;
`urlab.perception.ArucoDetector` pulls in cv2 only when you actually reach for it.
"""

_LAZY = {
    'RealSenseCamera': '.camera',
    'Frame': '.camera',
    'ArucoDetector': '.aruco',
    'MarkerTracker': '.aruco',
    'ConnectorEstimator': '.connector',
    'ConnectorTracker': '.connector',
    'CableReconstructor': '.cable_recon',
    'Reconstruction': '.cable_recon',
    # Junction geometry, vendored from sam3-abhay so it lives with the app. compute_junction
    # and the renderers need cv2; the graph tracer is numpy/scipy only, hence the split.
    'compute_junction': '.junction',
    'find_junction_index': '.junction',
    'render_overlay': '.junction',
    'save_profile_plot': '.junction',
    'trace_centerline_graph': '.cable_trace_graph',
    'trace_strands': '.cable_trace_graph',
    # Neck/tip geometry and the SAM3 segmentation wrapper, vendored alongside it. Sam3Backend is
    # the only thing here that reaches for torch, and it does so inside __init__, so naming it
    # costs nothing until it is built.
    'compute_necks': '.neck',
    'compute_tip': '.neck',
    'Sam3Backend': '.sam3_backend',
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
