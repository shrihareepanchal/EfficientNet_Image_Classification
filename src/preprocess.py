import os
import shutil

# 1. Define source and destination directories
# 'labeling' → original dataset (nested structure)
# 'data'     → cleaned dataset (brand-level structure)

src = 'labeling'
dst = 'data'


# 2. Remove existing processed data (if any)
# This ensures we always start fresh and avoid duplicate images
if os.path.exists(dst):
    shutil.rmtree(dst)

# Create new clean data directory
os.makedirs(dst)


# 3. Traverse through each brand folder
# Each top-level folder represents a brand
for brand in os.listdir(src):
    brand_path = os.path.join(src, brand)

    # Process only directories (ignore files)
    if os.path.isdir(brand_path):

        # Create a corresponding folder in 'data'
        # This will act as one class (brand-level classification)
        new_brand_path = os.path.join(dst, brand)
        os.makedirs(new_brand_path, exist_ok=True)


        # 4. Recursively collect all images under the brand
        # Using os.walk allows us to:
        # - Traverse subfolders like blanco, reposado, anejo
        # - Flatten structure into a single brand folder

        for root, dirs, files in os.walk(brand_path):
            for file in files:

                # 5. Filter only valid image formats
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):

                    # Full path of source image
                    src_file = os.path.join(root, file)

                    # Destination path (flattened into brand folder)
                    dst_file = os.path.join(new_brand_path, file)

                    # 6. Copy image to new structure
                    # shutil.copy is used instead of move:
                    # → preserves original dataset
                    shutil.copy(src_file, dst_file)
