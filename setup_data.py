import os
import shutil
import random

# Set the path to your data folders
data_path = "data/"

# Set the ratios for your train, validation, and test sets
train_size = 8600
val_size = 2000
test_size = 2000

# Set the paths to your train, validation, and test set folders
root = "data/stacked_pix2pix"

# Loop through each class (eevee and cycles) and split the data into train, validation, and test sets
for class_name in ["eevee", "cycles", "depth", "normal"]:
    # Get the list of files in the class folder
    class_files = sorted(os.listdir(os.path.join(data_path, class_name)))

    # Split the files into train, validation, and test sets
    train_files = class_files[:train_size]
    val_files = class_files[train_size:train_size+val_size]
    test_files = class_files[train_size+val_size:]
    # Copy the train files to the train set folder
    for file_name in train_files:
        src_path = os.path.join(data_path, class_name, file_name)
        dst_path = os.path.join(root, f"{class_name.upper()}/", "train", file_name)
        # print(f"Train")
        # print(src_path)
        # print(dst_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
    # Copy the validation files to the validation set folder
    for file_name in val_files:
        src_path = os.path.join(data_path, class_name, file_name)
        dst_path = os.path.join(root, f"{class_name.upper()}/", "val", file_name)
        # print(f"Val")
        # print(src_path)
        # print(dst_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
    # Copy the test files to the test set folder
    for file_name in test_files:
        src_path = os.path.join(data_path, class_name, file_name)
        dst_path = os.path.join(root, f"{class_name.upper()}/", "test", file_name)
        # print(f"Test")
        # print(src_path)
        # print(dst_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
