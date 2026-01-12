import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- CONFIGURATION ---
# Path to your best trained model
MODEL_PATH = 'best_mushroom_model.keras'
# Path to the image you want to test (Change this filename to test different images)
IMAGE_PATH = 'test_mantar.jpg' 

# 1. LOAD THE MODEL
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file '{MODEL_PATH}' not found. Please train the model first.")
else:
    print(f"--- [INFO] Loading Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. CHECK IMAGE EXISTENCE
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image '{IMAGE_PATH}' not found! Please check the filename.")
    else:
        # 3. PREPROCESS IMAGE
        # Target size must match the training input (224x224)
        img = image.load_img(IMAGE_PATH, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 224, 224, 3)
        img_array /= 255.0 # Normalize pixel values

        # 4. MAKE PREDICTION
        prediction = model.predict(img_array)
        confidence_score = prediction[0][0]

        # 5. INTERPRET RESULT
        # Class Indices: 0 -> Edible, 1 -> Poisonous
        print(f"\n--- ANALYSIS RESULT for {IMAGE_PATH} ---")
        
        if confidence_score < 0.5:
            # Result is closer to 0 (Edible)
            probability = (1 - confidence_score) * 100
            print(f"PREDICTION: EDIBLE (Yenilebilir)")
            print(f"CONFIDENCE: %{probability:.2f}")
        else:
            # Result is closer to 1 (Poisonous)
            probability = confidence_score * 100
            print(f"PREDICTION: POISONOUS (Zehirli)")
            print(f"CONFIDENCE: %{probability:.2f}")