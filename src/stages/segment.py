import numpy as np

from ultralytics import YOLO


class Segmentator:
    def __init__(
        self, model_type: str = 'yolov8n-seg',
        person_label: int = 0, moto_label: int = 3,
        conf_thr: float = 0.3, iou_thr: float = 0.7,
    ):
        self.model = YOLO(f'{model_type}.pt')

        self.person_label = person_label
        self.moto_label = moto_label
        self.consider_classes = [self.person_label, self.moto_label]

        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

    def segment(self, frame: np.array, verbose: bool = False):
        results = self.model.predict(
            frame, conf=self.conf_thr, iou=self.iou_thr,
            classes=self.consider_classes, verbose=verbose,
        )
        return results
