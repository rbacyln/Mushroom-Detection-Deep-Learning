import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration Parameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. DATA AUGMENTATION (Training Data)
train_datagen = ImageDataGenerator(
    rescale=1./255,           
    rotation_range=40,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    shear_range=0.2,          
    zoom_range=0.2,           
    horizontal_flip=True,     
    fill_mode='nearest'       
)

# 2. DATA NORMALIZATION (Test/Validation Data)
test_datagen = ImageDataGenerator(rescale=1./255)

print("\n--- [INFO] Data Loading and Preprocessing Started ---")

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load Test Data
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Class Indices: {train_generator.class_indices}")
print("--- [INFO] Data Preparation Completed ---\n")