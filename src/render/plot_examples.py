import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from src import Processing
from src.render import overlay, plot_one_box


def show_examples(
    result_function,
    processor: Processing,
    plot_label: str, images_folder: str,
    max_count: int = 12, cols: int = 3,
    img_size_on_plot: int = 5,
):
    rows = max_count // cols

    fig = plt.figure(
        figsize=(img_size_on_plot * rows, img_size_on_plot * cols),
    )
    fig.suptitle(plot_label)
    
    img_names = os.listdir(images_folder)
    k = min(max_count, len(img_names))
    example_imgs = random.sample(img_names, k)
    
    for i, img_name in enumerate(example_imgs):
        img_path = os.path.join(images_folder, img_name)
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
