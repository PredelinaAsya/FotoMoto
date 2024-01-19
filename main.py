import argparse
from typing import Literal

from src.pipeline import Processing


if __name__ == '__main__':
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
