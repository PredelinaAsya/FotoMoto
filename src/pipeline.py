import cv2
from multiprocessing import Pool
import numpy as np
import os
import rawpy
from shutil import copy2
from typing import Tuple, Literal

from src.stages import (
    Segmentator,
    match_motorcycles_and_pilots,
    compute_embedding_by_separate_channels,
    compute_embedding_by_union_channels,
    cluster_with_kmeans,
    )


class Processing:
    def __init__(
        self, images_folder: str,
        results_folder: str = 'results',
        support_img_formats: Tuple[str] = ('jpg', 'png', 'cr2'),
        model_type: str = 'yolov8n-seg',
        person_label: int = 0, moto_label: int = 3,
        conf_thr: float = 0.2, iou_thr: float = 0.65,
        first_part_processes: int = 4,
        hsv_flag: bool = True, intervals_count: int = 256,
        embedding_type: Literal['separate', 'union'] = 'separate',
        k_min: int = 10, k_max: int = 100,
    ):
        if not os.path.exists(images_folder):
            raise ValueError(f'Input folder: {images_folder} does not exist')

        self.support_img_formats = support_img_formats

        self.img_paths = [
            os.path.join(images_folder, img_name)
            for img_name in os.listdir(images_folder)
            if img_name.lower().endswith(self.support_img_formats)
        ]

        if not os.path.exists(results_folder):
            os.makedirs(results_folder, exist_ok=True)

        self.results_folder = results_folder

        self.segmentator = Segmentator(
            model_type=model_type, person_label=person_label,
            moto_label=moto_label, conf_thr=conf_thr,
            iou_thr=iou_thr,
        )

        self.first_part_processes = first_part_processes
        self.hsv_flag = hsv_flag
        self.intervals_count = intervals_count
        self.embedding_type = embedding_type
        self.min_k = k_min
        self.max_k = k_max

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
            
        return processed_masks, image

    def compute_color_embeddings_by_separate_channels_on_image(
        self, rgb_img, moto_masks,
    ):  
        color_embs = []

        for moto_mask in moto_masks:
            color_embedding = compute_embedding_by_separate_channels(
                rgb_img, moto_mask, hsv_flag=self.hsv_flag,
                intervals_count=self.intervals_count,
            )
            color_embs.append(color_embedding)

        return color_embs
    
    def compute_color_embeddings_by_union_channels_on_image(
        self, rgb_img, moto_masks,
    ):  
        color_embs = []

        for moto_mask in moto_masks:
            color_embedding = compute_embedding_by_union_channels(
                rgb_img, moto_mask, hsv_flag=self.hsv_flag,
                out_color_dim=self.intervals_count,
            )
            color_embs.append(color_embedding)

        return color_embs
    
    def compute_embs_process(self, img_path):
        processed_masks, image = self.get_processed_masks_on_image(img_path)

        if self.embedding_type == 'separate':
            color_embs = self.compute_color_embeddings_by_separate_channels_on_image(
                image, processed_masks,
            )
        else:
            color_embs = self.compute_color_embeddings_by_union_channels_on_image(
                image, processed_masks,
            )

        return color_embs

    def process_all_images(self, processes: int = 4):
        moto_embs = []

        with Pool(processes) as pool:
            moto_embs = pool.map(self.compute_embs_process, self.img_paths)

        img_path_to_embs_and_masks = {
            img_path:embs for img_path, embs in zip(self.img_paths, moto_embs)
        }

        return img_path_to_embs_and_masks
    
    def cluster_embeddings(self, img_path_to_moto_embs):
        all_embs = []
        img_paths = []

        for img_path, embs in img_path_to_moto_embs.items():
            all_embs.extend(embs)
            img_paths.extend([img_path] * len(embs))

        X = np.array(all_embs)

        cluster_id_to_elems = cluster_with_kmeans(X, min_k=self.min_k, max_k=self.max_k)
        cluster_id_to_img_paths = []
        
        for cluster_elems in cluster_id_to_elems:
            cluster_img_paths = [img_paths[elem_id] for elem_id in cluster_elems]
            cluster_id_to_img_paths.append(cluster_img_paths)

        return cluster_id_to_img_paths
    
    def save_cluster_images_in_different_folders(self, cluster_id_to_img_paths):
        for cluster_id, src_img_paths in cluster_id_to_img_paths.items():
            folder_name = f'folder_{cluster_id + 1}'
            folder_path = os.path.join(self.results_folder, folder_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            for img_path in src_img_paths:
                new_img_path = os.path.join(folder_path, os.path.basename(img_path))
                copy2(img_path, new_img_path)

    def pipeline(self):
        img_path_to_moto_embs = self.process_all_images(
            processes=self.first_part_processes
        )
        cluster_id_to_img_paths = self.cluster_embeddings(img_path_to_moto_embs)
        self.save_cluster_images_in_different_folders(cluster_id_to_img_paths)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--images_folder', type=str, nargs=1,
        help='Input folder with photos from motorace (all images should be in jpg, png or cr2 format)',
    )
    parser.add_argument(
        '--results_folder', type=str, default='results',
        help='Directory to save sorting fotos into different folders',
    )
    parser.add_argument(
        '--first_part_processes', type=int, default=4,
        help="Number of parallel processes for first algorithm stage (segmentation + compute embeddings)",
    )
    parser.add_argument(
        '--hsv_flag', type=bool, default=False,
        help='Convert image from RGB to HSV or not before calculate embeddings',
    )
    parser.add_argument(
        '--intervals_count', type=int, default=8,
        help='What color dimension use in color embeddings',
    )
    parser.add_argument(
        '--embedding_type', type=Literal['separate', 'union'], default='union',
        help='Type of embedding calculation: `separate` or `union` channels',
    )
    parser.add_argument(
        '--k_min', type=int, default=10,
        help='Minimum available value for count of clusters',
    )
    parser.add_argument(
        '--k_max', type=int, default=100,
        help='Maximum available value for count of clusters',
    )

    opt = parser.parse_args()

    processing = Processing(**vars(opt))
    processing.pipeline()
