import os
import shutil
import random

# Path to your dataset
DATA_DIR = "data/PlantVillage"
OUTPUT_DIR = "data"

# Train/Val/Test split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output folders
for split in ["train", "val", "test"]:
    split_path = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_path, exist_ok=True)

# Loop over each disease class
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Copy files into new folders
    for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        split_class_path = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_class_path, exist_ok=True)
        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(split_class_path, file)
            shutil.copy(src, dst)

print("✅ Dataset split completed! Check the 'data/train', 'data/val', and 'data/test' folders.")
