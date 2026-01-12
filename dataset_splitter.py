import os
import shutil
import random

# --- CONFIGURATION ---
# Path definitions based on your project structure
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
CLASSES = ["Edible", "Poisonous"]

def split_dataset(split_ratio=0.2):
    """
    Splits the dataset into Training and Test sets based on the given ratio.
    Moves files from TRAIN_DIR to TEST_DIR.
    """
    print("--- [INFO] Dataset Splitting Process Started ---")
    
    for class_name in CLASSES:
        source_dir = os.path.join(TRAIN_DIR, class_name)
        target_dir = os.path.join(TEST_DIR, class_name)
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # List only valid image files
        images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_count = len(images)
        
        # Calculate how many images to move to Test
        test_count = int(total_count * split_ratio)
        
        print(f"\nProcessing Class: {class_name}")
        print(f"Total Images Found: {total_count}")
        
        if total_count == 0:
            print(f"Warning: No images found in {class_name}. Skipping...")
            continue
        
        # Shuffle to ensure random selection
        random.shuffle(images)
        test_images = images[:test_count]
        
        print(f"Moving {test_count} images to Test folder ({split_ratio*100}%)...")
        
        # Move files
        for img in test_images:
            src_path = os.path.join(source_dir, img)
            dst_path = os.path.join(target_dir, img)
            shutil.move(src_path, dst_path)
            
        print(f"Images remaining in Train: {total_count - test_count}")

    print("\n--- [SUCCESS] Dataset Splitting Completed ---")

if __name__ == "__main__":
    # WARNING: Run this only once! Running it multiple times will keep reducing your training set.
    split_dataset(split_ratio=0.2)