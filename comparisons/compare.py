# SSIM and MAE Error

import torch  
import torch.nn.functional as F 
import numpy as np
import math
from PIL import Image
import cv2
from pathlib import Path
import re
import os
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
import sys
import pandas as pd

DATASETS = []
# DATASETS.append("stacked_depth_full")
# DATASETS.append("milestone_results")
# DATASETS.append("milestone_model_train")
DATASETS.append("new_data_base")
DATASETS.append("new_data_stacked_depth")
DATASETS.append("new_data_stacked_depth_skip")
DATASETS.append("new_data_stacked_depth_normal")
DATASETS.append("new_data_stacked_depth_deep")

USER = "Caroline"

SIZE = 256
WINDOW_SIZE = 11 # default for ssim
SIGMA = 1.5 # default for ssim
C = 3
NORMALIZE_MAE = False

def load_images(x):
    return np.asarray(Image.open(x).resize((SIZE, SIZE)))

def ssim_wrapper(im1, im2):
    # expects uint8 input
    return ssim(im1, im2, channel_axis=2)

def mae(im1, im2):
    # Calculate the squared difference between the pixel values for each channel; scaled btw 0 and 1
    if NORMALIZE_MAE:
        scaled_im1 = im1.astype(np.float64) / 255
        scaled_im2 = im2.astype(np.float64) / 255

        abs_diff = np.abs(scaled_im1 - scaled_im2)
    
    else:
        abs_diff = np.abs(im1.astype(np.float64) - im2.astype(np.float64))

    # Calculate the mean of the squared differences across all channels
    return np.mean(abs_diff)

def main():
    data = dict()
    data["dataset"] = []
    data["ssim"] = []
    data["mae"] = []

    for d in DATASETS:
        if USER == "Caroline":
            if d == "local_results":
                PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/local_results/ray_pix2pix/test_latest/images/"
            
            elif d == "stacked_depth_full":
                PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/stacked_depth_full/test_latest/images/"
            
            elif d == "milestone_results":
                PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/milestone_results/full_train_ray/test_latest/images/"
            
            elif d == "milestone_model_train":
                PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/milestone_model_train/test_latest/images/"

            elif d == "new_data_base":
                PATH_TO_RESULTS ="/Users/carolinecahilly/Desktop/results/new_data_base/test_latest/images/"

            elif d == "new_data_stacked_depth":
                PATH_TO_RESULTS ="/Users/carolinecahilly/Desktop/results/new_data_stacked_depth/test_latest/images/"

            elif d == "new_data_stacked_depth_skip":
                PATH_TO_RESULTS ="/Users/carolinecahilly/Desktop/results/new_data_stacked_depth_skip/test_latest/images/"

            elif d == "new_data_stacked_depth_normal":
                PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/new_data_stacked_depth_normal/test_latest/images/"
            
            elif d == "new_data_stacked_depth_deep":
                PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/new_data_stacked_depth_deep/test_latest/images/"

            else:
                raise ValueError("Enter a valid dataset name")

            # OUTPUT_FOLDER = d + "/"

        mug_views = set()
        for item in Path(PATH_TO_RESULTS).iterdir():
            # Use regular expression to match the desired substring
            match = re.search(r"(.*_)\w+_", str(item.relative_to(PATH_TO_RESULTS)))

            if match:
                result = match.group(1)
            
            mug_views.add(result)
        
        avg_ssim_true_vs_false = 0
        avg_mae_true_vs_false = 0
        for view in mug_views:
            # The true reference Image 
            true_img = load_images(PATH_TO_RESULTS + view + "real_B.png")

            # The False image
            false_img = load_images(PATH_TO_RESULTS + view + "fake_B.png")

            # True image vs False Image
            avg_ssim_true_vs_false += ssim_wrapper(true_img, false_img)
            avg_mae_true_vs_false += mae(true_img, false_img)

        avg_ssim_true_vs_false /= len(mug_views)
        avg_mae_true_vs_false /= len(mug_views)

        avg_ssim_true_vs_false = round(avg_ssim_true_vs_false.item(), 3)
        avg_mae_true_vs_false = round(avg_mae_true_vs_false.item(), 3)

        data["dataset"].append(d)
        data["ssim"].append(avg_ssim_true_vs_false)
        data["mae"].append(avg_mae_true_vs_false)
        print("\n" + d + "\n")
        print("SSIM")
        print("Window_size: ", str(WINDOW_SIZE))
        print("Sigma: ", str(SIGMA))
        print("The average ssim score for true vs. false: " + str(avg_ssim_true_vs_false))
        print("\nMAE")
        print("The average mae score for true vs. false: " + str(avg_mae_true_vs_false))

    df = pd.DataFrame(data)
    df.to_excel("eval_results.xlsx", index=False)

if __name__ == "__main__":
    main()