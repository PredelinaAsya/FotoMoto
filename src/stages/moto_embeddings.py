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


def compute_embedding_by_union_channels(
    rgb_img, moto_mask, hsv_flag=True,
    out_color_dim=16, input_color_dim=256,
):
    image = rgb_img.copy()
    if hsv_flag:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    divider = input_color_dim // out_color_dim
    image = np.floor(np.divide(image, divider).astype(int))
    
    args = np.argwhere(moto_mask[:, :, 0])
    xpos = image[args, 0].astype(np.float32).reshape(-1)
    ypos = image[args, 1].astype(np.float32).reshape(-1)
    zpos = image[args, 2].astype(np.float32).reshape(-1)
    
    encoded_triplets_arr = xpos * out_color_dim ** 2 + ypos * out_color_dim + zpos
    emb_dim = out_color_dim ** 3
    
    pixels_in_mask = moto_mask[:, :, 0].sum()

    H, _ = np.histogram(
        encoded_triplets_arr, bins=emb_dim-1, range=(0,emb_dim-1),
        density=False
    )
    
    color_embedding = H.astype(np.float32) / pixels_in_mask
        
    return color_embedding
