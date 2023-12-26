import cv2
import numpy as np
import os
import rawpy
from typing import Tuple

from src.stages import Segmentator, match_motorcycles_and_pilots


class Processing:
    def __init__(
        self, images_folder: str,
        support_img_formats: Tuple[str] = ('jpg', 'png', 'cr2'),
        model_type: str = 'yolov8n-seg',
        person_label: int = 0, moto_label: int = 3,
        conf_thr: float = 0.2, iou_thr: float = 0.65,
    ):
        if not os.path.exists(images_folder):
            raise ValueError(f'Input folder: {images_folder} does not exist')

        self.support_img_formats = support_img_formats

        self.img_paths = [
            os.path.join(images_folder, img_name)
            for img_name in os.listdir(images_folder)
            if img_name.lower().endswith(self.support_img_formats)
        ]

        self.segmentator = Segmentator(
            model_type=model_type, person_label=person_label,
            moto_label=moto_label, conf_thr=conf_thr,
            iou_thr=iou_thr,
        )

    def get_moto_masks_on_image(self, img_path: str):
        if not os.path.exists(img_path):
            raise ValueError(f'Input image path: {img_path} does not exist')

        if not img_path.lower().endswith(self.support_img_formats):
            raise ValueError(
                f'Input image: {img_path} has unsupported format. List of supported formats: {self.support_img_formats}')
        
        # read image
        if img_path.lower().endswith('.cr2'):
            raw = rawpy.imread(img_path) # access to the RAW image
            image = raw.postprocess() # a numpy RGB array
        else:
            image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # segment motorcycles and pilots
        segment_results = self.segmentator.segment(image)[0]
        person_boxes = [
            box.data[0] for box in segment_results.boxes
            if box.cls == self.segmentator.person_label
        ]
        moto_boxes = [
            box.data[0] for box in segment_results.boxes
            if box.cls == self.segmentator.moto_label
        ]

        # matching between motorcycles and pilots
        matched_moto_to_pilots = match_motorcycles_and_pilots(person_boxes, moto_boxes)

        return segment_results, matched_moto_to_pilots, image
    
    def get_processed_masks_on_image(self, img_path: str):
        segment_results, matched_moto_to_pilots, image = self.get_moto_masks_on_image(img_path)
        h, w, _ = image.shape

        person_ids = [
            i for i, box in enumerate(segment_results.boxes)
            if box.cls == self.segmentator.person_label
        ]
        moto_ids = [
            i for i, box in enumerate(segment_results.boxes)
            if box.cls == self.segmentator.moto_label
        ]
        
        masks = segment_results.masks
        boxes = segment_results.boxes

        processed_masks = []

        if masks is not None:
            masks = masks.data.cpu()

            for moto_bbox_id, person_bbox_ids in matched_moto_to_pilots.items():
                det_ids = [moto_ids[moto_bbox_id]]
                det_ids.extend([person_ids[person_bbox_id] for person_bbox_id in person_bbox_ids])

                for idx in det_ids:
                    seg, box = masks.numpy()[idx], boxes[idx]
                        
                    seg = cv2.resize(seg, (w, h))
                    colored_mask = np.expand_dims(seg, 0).repeat(3, axis=0)
                    colored_mask = np.moveaxis(colored_mask, 0, -1)

                    processed_masks.append(np.round(colored_mask))
            
        return processed_masks
