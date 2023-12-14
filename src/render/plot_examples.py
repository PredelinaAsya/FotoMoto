import os
import cv2
import numpy as np
import random
from typing import Union, List
import matplotlib.pyplot as plt

from src import Processing
from src.render import overlay, plot_one_box


def show_examples(
    result_function,
    processor: Processing,
    plot_label: str, images_folder: Union[str, None],
    img_paths: Union[List[str], None] = None,
    max_count: int = 12, cols: int = 3,
    img_size_on_plot: int = 5,
):
    rows = max_count // cols

    fig = plt.figure(
        figsize=(img_size_on_plot * rows, img_size_on_plot * cols),
    )
    fig.suptitle(plot_label)
    
    if images_folder is not None:
        img_names = os.listdir(images_folder)
    else:
        img_names = img_paths
    
    k = min(max_count, len(img_names))
    example_imgs = random.sample(img_names, k)
    
    for i, img_name in enumerate(example_imgs):
        if images_folder is not None:
            img_path = os.path.join(images_folder, img_name)
        else:
            img_path = img_name

        rendered_img = result_function(img_path, processor)
        
        fig.add_subplot(rows, cols, i+1)
        plt.axis('off')
        plt.imshow(rendered_img)
    
    plt.show()


def segment_moto_on_image(img_path: str, processor: Processing):
    colors = {
        processor.segmentator.person_label: [255, 0, 255],
        processor.segmentator.moto_label: [255, 255, 0],
    }

    segment_results, _, image = processor.get_moto_masks_on_image(img_path)
    h, w, _ = image.shape
    frame_with_results = image.copy()

    for r in segment_results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):

                seg = cv2.resize(seg, (w, h))
                frame_with_results = overlay(frame_with_results, seg, colors[int(box.cls)], 0.4)

                xmin = int(box.data[0][0])
                ymin = int(box.data[0][1])
                xmax = int(box.data[0][2])
                ymax = int(box.data[0][3])

                plot_one_box(
                    [xmin, ymin, xmax, ymax], frame_with_results, colors[int(box.cls)],
                    f'{float(box.conf):.3}',
                )
        
    return frame_with_results


def segment_moto_and_match_masks_on_image(
    img_path: str, processor: Processing,
):
    colors = [
        [255, 0, 255], [255, 255, 0], [255, 0, 0],
        [0, 255, 0], [0, 0, 255], [0, 255, 255]
    ]

    segment_results, matched_moto_to_pilots, image = processor.get_moto_masks_on_image(img_path)
    h, w, _ = image.shape
    frame_with_results = image.copy()
    
    person_boxes = [
        box.data[0] for box in segment_results.boxes
        if box.cls == processor.segmentator.person_label
    ]
    moto_boxes = [
        box.data[0] for box in segment_results.boxes
        if box.cls == processor.segmentator.moto_label
    ]
    
    person_ids = [
        i for i, box in enumerate(segment_results.boxes)
        if box.cls == processor.segmentator.person_label
    ]
    moto_ids = [
        i for i, box in enumerate(segment_results.boxes)
        if box.cls == processor.segmentator.moto_label
    ]
    
    masks = segment_results.masks
    boxes = segment_results.boxes

    if masks is not None:
        masks = masks.data.cpu()

        for moto_bbox_id, person_bbox_ids in matched_moto_to_pilots.items():
            color = colors[int(moto_bbox_id)]
            det_ids = [moto_ids[moto_bbox_id]]
            det_ids.extend([person_ids[person_bbox_id] for person_bbox_id in person_bbox_ids])

            for idx in det_ids:
                seg, box = masks.numpy()[idx], boxes[idx]
                    
                seg = cv2.resize(seg, (w, h))
                frame_with_results = overlay(frame_with_results, seg, color, 0.4)
        
    return frame_with_results
