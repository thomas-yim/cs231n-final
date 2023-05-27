import os
from PIL import Image

# Set the directory you want to start from
root_dir = '/Users/thomasyim/Dropbox/mac/Stanford/Sophomore/Spring/CS231N/final/data'
count = 0
for dir_name, subdir_list, file_list in os.walk(root_dir):
    for fname in file_list:
        if fname.endswith('.png'):
            img_path = os.path.join(dir_name, fname)
            img = Image.open(img_path)
            
            # Check if image is RGBA
            if img.mode == 'RGBA':
                # Convert the image from RGBA to RGB
                rgb_img = img.convert('RGB')
                # Save the image
                rgb_img.save(img_path)
                print(f"Converted {img_path} to RGB")
                count += 1

print(count)
