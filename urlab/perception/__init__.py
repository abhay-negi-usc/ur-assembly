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
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
