# Code pulled from https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e

import torch  
import torch.nn.functional as F 
import numpy as np
import math
from PIL import Image
import cv2
from pathlib import Path
import re
import os

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

    OUTPUT_FOLDER = DATASET + "/"

SIZE = 256
WINDOW_SIZE = 11
SIGMA = 1.5
C = 3
SAVE_IMAGES = False

def gaussian(window_size=WINDOW_SIZE, sigma=SIGMA):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size=WINDOW_SIZE, channel=C):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size).unsqueeze(1)
    
    # Converting to 2D  
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def ssim(img1, img2, val_range, window_size=WINDOW_SIZE, window=None, size_average=True, full=False):

    L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be at least WINDOW_SIZE x WINDOW_SIZE 
        window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    return ret

def load_images(x):
    return np.asarray(Image.open(x).resize((SIZE, SIZE)))

def tensorify(x):
    return torch.Tensor(x.transpose((2, 0, 1)).copy()).unsqueeze(0).float().div(255.0)

# save imgs 
def save_imgs(x, p, transpose=True, resize=True):
    if resize:
        x=cv2.resize(x, (SIZE, SIZE))
    if transpose:
        x_transpose = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        # norm = cv2.normalize(x_transpose, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        Image.fromarray(x_transpose).save(p)
    else:
        # norm = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        Image.fromarray(x).save(p)

def main():
    # gauss_dis = gaussian()
    # print("Distribution: ", gauss_dis)
    # print("Sum of Gauss Distribution:", torch.sum(gauss_dis))

    # window = create_window()
    # print("Shape of gaussian window:", window.shape)

    mug_views = set()
    for item in Path(PATH_TO_RESULTS).iterdir():
        # Use regular expression to match the desired substring
        match = re.search(r"(.*_)\w+_", str(item.relative_to(PATH_TO_RESULTS)))

        if match:
            result = match.group(1)
        
        mug_views.add(result)
    
    scores = dict()
    avg_true_vs_false = 0
    avg_true_vs_noised = 0
    avg_true_vs_true = 0
    for view in mug_views:
        # The true reference Image 
        img1 = load_images(PATH_TO_RESULTS + view + "real_B.png")

        # The False image
        img2 = load_images(PATH_TO_RESULTS + view + "fake_B.png")

        # The noised true image
        noise = np.random.randint(0, 255, (SIZE, SIZE, 3)).astype(np.float32)
        noisy_img_unnormalized = img1 + noise
        noisy_img = cv2.normalize(noisy_img_unnormalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if SAVE_IMAGES:
            images_folder = OUTPUT_FOLDER + "images/"
            if not os.path.exists(images_folder):
                os.makedirs(images_folder)

            save_imgs(img1, images_folder + view + "true_img.png")
            save_imgs(img2, images_folder + view + "false_img.png")
            save_imgs(noisy_img, images_folder + view + "noised_img.png")

        # Check SSIM score of True image vs False Image
        _img1 = tensorify(img1)
        _img2 = tensorify(img2)
        true_vs_false = ssim(_img1, _img2, val_range=255)
        # print("True vs False Image SSIM Score:", true_vs_false)
        scores[view + "true_vs_false"] = true_vs_false
        avg_true_vs_false += true_vs_false

        # Check SSIM score of True image vs Noised_true Image
        _img1 = tensorify(img1)
        _img2 = tensorify(noisy_img)
        true_vs_false = ssim(_img1, _img2, val_range=255)
        # print("True vs Noisy True Image SSIM Score:", true_vs_false)
        scores[view + "true_vs_noised"] = true_vs_false
        avg_true_vs_noised += true_vs_false

        # Check SSIM score of True image vs True Image
        _img1 = tensorify(img1)
        true_vs_false = ssim(_img1, _img1, val_range=255)
        # print("True vs True Image SSIM Score:", true_vs_false)
        scores[view + "true_vs_true"] = true_vs_false
        avg_true_vs_true += true_vs_false

    avg_true_vs_false /= len(mug_views)
    avg_true_vs_noised /= len(mug_views)
    avg_true_vs_true /= len(mug_views)

    if not SAVE_IMAGES:
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

    with open(OUTPUT_FOLDER + "results.txt", "w") as file:
        # Redirect the output to the file
        print("Window_size: ", str(WINDOW_SIZE), file=file)
        print("Sigma: ", str(SIGMA), file=file)
        print("The average ssim score for true vs. false: " + str(round(avg_true_vs_false.item(), 3)), file=file)
        print("The average ssim score for true vs. noised: " + str(round(avg_true_vs_noised.item(), 3)), file=file)
        print("The average ssim score for true vs. true: " + str(round(avg_true_vs_true.item(), 3)), file=file)

    print("Window_size: ", str(WINDOW_SIZE))
    print("Sigma: ", str(SIGMA))
    print("The average ssim score for true vs. false: " + str(round(avg_true_vs_false.item(), 3)))
    print("The average ssim score for true vs. noised: " + str(round(avg_true_vs_noised.item(), 3)))
    print("The average ssim score for true vs. true: " + str(round(avg_true_vs_true.item(), 3)))

    

main()