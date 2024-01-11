import cv2
import numpy as np


def compute_embedding_by_separate_channels(
    rgb_img, moto_mask, hsv_flag=True,
    intervals_count=256
):
    image = rgb_img.copy()
    if hsv_flag:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
    _, _, channels = image.shape
    
    color_embedding = np.zeros(intervals_count * channels)
    
    for channel_id in range(channels):
        H, _ = np.histogram(
            image[:, :, channel_id], bins=intervals_count - 1, range=(0,255),
            density=False, weights=moto_mask[:, :, channel_id]
        )
        pixels_in_mask = moto_mask[:, :, channel_id].sum()
        color_embedding[channel_id * intervals_count : (channel_id + 1) * intervals_count - 1] = H.astype(np.float32) / pixels_in_mask
        
    return color_embedding
