# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, pose, segment

from .model import YOLO
from .model import YOLO_zhr

__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO'
