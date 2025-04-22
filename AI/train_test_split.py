import os
import random
import shutil

# Define your dataset directories
image_dir = './dataset/images'
annotation_dir = './dataset/converted_labels'
train_dir = './dataset/train'
val_dir = './dataset/val'
test_dir = './dataset/test'

# Ensure output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all the image files in the image directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Function to check if the image contains the category (based on annotation)
def contains_category(image_file, category_id=0):  # default is category_id=0
    annotation_file = os.path.join(annotation_dir, image_file.replace('.jpg', '.txt'))
    if not os.path.exists(annotation_file):
        return False
    
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) == category_id:
                return True
    return False

# Filter the images to include only those containing the category of interest
category_images = [f for f in image_files if contains_category(f)]

# Shuffle the list for random splitting
random.shuffle(category_images)

# Calculate the number of images for each split
train_size = int(0.7 * len(category_images))
val_size = int(0.15 * len(category_images))
test_size = len(category_images) - train_size - val_size

# Split the images into train, val, and test
train_images = category_images[:train_size]
val_images = category_images[train_size:train_size+val_size]
test_images = category_images[train_size+val_size:]

# Helper function to copy images and annotations to the destination folder
def copy_files(file_list, dest_dir):
    for image_file in file_list:
        # Copy image
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(dest_dir, 'images', image_file))
        # Copy annotation
        annotation_file = image_file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(annotation_dir, annotation_file), os.path.join(dest_dir, 'labels', annotation_file))

# Create subdirectories for images and labels in each set
for subset in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(subset, 'images'), exist_ok=True)
    os.makedirs(os.path.join(subset, 'labels'), exist_ok=True)

# Copy files to their respective directories
copy_files(train_images, train_dir)
copy_files(val_images, val_dir)
copy_files(test_images, test_dir)

print(f"Dataset split completed: {len(train_images)} for training, {len(val_images)} for validation, {len(test_images)} for testing.")
