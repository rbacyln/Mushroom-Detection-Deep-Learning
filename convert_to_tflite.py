import tensorflow as tf
import os

# --- CONFIGURATION ---
MODEL_PATH = 'best_mushroom_model.keras'
TFLITE_PATH = 'mushroom_model_lite.tflite'

# 1. LOAD THE TRAINED KERAS MODEL
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file '{MODEL_PATH}' not found!")
    exit()

print(f"--- [INFO] Loading model from {MODEL_PATH} ---")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. CONVERT TO TFLITE
print("--- [INFO] Starting TFLite conversion (This may take a moment)... ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 3. SAVE THE TFLITE MODEL
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"\n--- [SUCCESS] Model converted and saved as '{TFLITE_PATH}' ---")
kb_size = os.path.getsize(TFLITE_PATH) / 1024
print(f"Final Mobile Model Size: {kb_size:.2f} KB")