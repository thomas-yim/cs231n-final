# SSIM, PSNR, and MSE Error

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
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import defaultdict
import sys

if len(sys.argv) > 1:
    DATASET = sys.argv[1]
else:
    DATASET = "local_results"
    # DATASET = "stacked_depth_full"
    # DATASET = "milestone_results"
    # DATASET = "milestone_model_train"

USER = "Caroline"
if USER == "Caroline":
    if DATASET == "local_results":
        PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/local_results/ray_pix2pix/test_latest/images/"
    
    elif DATASET == "stacked_depth_full":
        PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/stacked_depth_full/test_latest/images/"
    
    elif DATASET == "milestone_results":
        PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/milestone_results/full_train_ray/test_latest/images/"
    
    elif DATASET == "milestone_model_train":
        PATH_TO_RESULTS = "/Users/carolinecahilly/Desktop/results/milestone_model_train/test_latest/images/"
    
    else:
        raise ValueError("Enter a valid dataset name")

    OUTPUT_FOLDER = DATASET + "/"

SIZE = 256
WINDOW_SIZE = 11 # default for ssim
SIGMA = 1.5 # default for ssim
C = 3
SAVE_IMAGES = False

def load_images(x):
    return np.asarray(Image.open(x).resize((SIZE, SIZE)))

def psnr_wrapper(im1, im2):
    if np.array_equal(im1, im2):
        return np.array([np.inf])
    
    return psnr(im1, im2, data_range=255)

def ssim_wrapper(im1, im2):
    return ssim(im1, im2, data_range=255, channel_axis=2)

def mse(im1, im2):
    # Calculate the squared difference between the pixel values for each channel
    squared_diff = (im1 - im2) ** 2

    # Calculate the mean of the squared differences across all channels
    return np.mean(squared_diff)

# save imgs 
def save_imgs(x, p, transpose=True, resize=True):
    if resize:
        x=cv2.resize(x, (SIZE, SIZE))
    if transpose:
        x_transpose = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        norm = cv2.normalize(x_transpose, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        Image.fromarray(norm).save(p)
    else:
        norm = np.zeros_like(x)
        norm = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        Image.fromarray(norm).save(p)

def main():
    mug_views = set()
    for item in Path(PATH_TO_RESULTS).iterdir():
        # Use regular expression to match the desired substring
        match = re.search(r"(.*_)\w+_", str(item.relative_to(PATH_TO_RESULTS)))

        if match:
            result = match.group(1)
        
        mug_views.add(result)
    
    avg_ssim_true_vs_false = 0
    avg_ssim_true_vs_noised = 0
    avg_ssim_true_vs_true = 0
    avg_psnr_true_vs_false = 0
    avg_psnr_true_vs_noised = 0
    avg_psnr_true_vs_true = 0
    avg_mse_true_vs_false = 0
    avg_mse_true_vs_noised = 0
    avg_mse_true_vs_true = 0
    for view in mug_views:
        # The true reference Image 
        true_img = load_images(PATH_TO_RESULTS + view + "real_B.png")

        # The False image
        false_img = load_images(PATH_TO_RESULTS + view + "fake_B.png")

        # The noised true image
        noise = np.random.randint(0, 255, (SIZE, SIZE, 3)).astype(np.float32)
        noisy_img_unnormalized = true_img + noise
        noisy_img = cv2.normalize(noisy_img_unnormalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if SAVE_IMAGES:
            images_folder = OUTPUT_FOLDER + "images/"
            if not os.path.exists(images_folder):
                os.makedirs(images_folder)

            save_imgs(true_img, images_folder + view + "true_img.png")
            save_imgs(false_img, images_folder + view + "false_img.png")
            save_imgs(noisy_img, images_folder + view + "noised_img.png")

        # True image vs False Image
        avg_ssim_true_vs_false += ssim_wrapper(true_img, false_img)
        avg_psnr_true_vs_false += psnr_wrapper(true_img, false_img)
        avg_mse_true_vs_false += mse(true_img, false_img)

        # True image vs Noised_true Image
        avg_ssim_true_vs_noised += ssim_wrapper(true_img, noisy_img)
        avg_psnr_true_vs_noised += psnr_wrapper(true_img, noisy_img)
        avg_mse_true_vs_noised += mse(true_img, noisy_img)

        # True image vs True Image
        avg_ssim_true_vs_true += ssim_wrapper(true_img, true_img)
        avg_psnr_true_vs_true += psnr_wrapper(true_img, true_img)
        avg_mse_true_vs_true += mse(true_img, true_img)

    avg_ssim_true_vs_false /= len(mug_views)
    avg_ssim_true_vs_noised /= len(mug_views)
    avg_ssim_true_vs_true /= len(mug_views)

    avg_psnr_true_vs_false /= len(mug_views)
    avg_psnr_true_vs_noised /= len(mug_views)
    avg_psnr_true_vs_true /= len(mug_views)

    avg_mse_true_vs_false /= len(mug_views)
    avg_mse_true_vs_noised /= len(mug_views)
    avg_mse_true_vs_true /= len(mug_views)

    if not SAVE_IMAGES:
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

    with open(OUTPUT_FOLDER + "complete_" + DATASET + "_results.txt", "w") as file:
        # Redirect the output to the file
        print("SSIM", file=file)
        print("Window_size: ", str(WINDOW_SIZE), file=file)
        print("Sigma: ", str(SIGMA), file=file)
        print("The average ssim score for true vs. false: " + str(round(avg_ssim_true_vs_false.item(), 3)), file=file)
        print("The average ssim score for true vs. noised: " + str(round(avg_ssim_true_vs_noised.item(), 3)), file=file)
        print("The average ssim score for true vs. true: " + str(round(avg_ssim_true_vs_true.item(), 3)), file=file)
        print("\nPSNR", file=file)
        print("The average psnr score for true vs. false: " + str(round(avg_psnr_true_vs_false.item(), 3)), file=file)
        print("The average psnr score for true vs. noised: " + str(round(avg_psnr_true_vs_noised.item(), 3)), file=file)
        print("The average psnr score for true vs. true: " + str(round(avg_psnr_true_vs_true.item(), 3)), file=file)
        print("\nMSE", file=file)
        print("The average mse score for true vs. false: " + str(round(avg_mse_true_vs_false.item(), 3)), file=file)
        print("The average mse score for true vs. noised: " + str(round(avg_mse_true_vs_noised.item(), 3)), file=file)
        print("The average mse score for true vs. true: " + str(round(avg_mse_true_vs_true.item(), 3)), file=file)

    print("SSIM")
    print("Window_size: ", str(WINDOW_SIZE))
    print("Sigma: ", str(SIGMA))
    print("The average ssim score for true vs. false: " + str(round(avg_ssim_true_vs_false.item(), 3)))
    print("The average ssim score for true vs. noised: " + str(round(avg_ssim_true_vs_noised.item(), 3)))
    print("The average ssim score for true vs. true: " + str(round(avg_ssim_true_vs_true.item(), 3)))
    print("\nPSNR")
    print("The average psnr score for true vs. false: " + str(round(avg_psnr_true_vs_false.item(), 3)))
    print("The average psnr score for true vs. noised: " + str(round(avg_psnr_true_vs_noised.item(), 3)))
    print("The average psnr score for true vs. true: " + str(round(avg_psnr_true_vs_true.item(), 3)))
    print("\nMSE")
    print("The average mse score for true vs. false: " + str(round(avg_mse_true_vs_false.item(), 3)))
    print("The average mse score for true vs. noised: " + str(round(avg_mse_true_vs_noised.item(), 3)))
    print("The average mse score for true vs. true: " + str(round(avg_mse_true_vs_true.item(), 3)))

main()