import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models
# Import data from our standardized loader
from data_loader import train_generator, test_generator

# 1. DEFINE BASELINE ARCHITECTURE (Transfer Learning)
# We use MobileNetV3Small as a lightweight baseline model to compare performance.
base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze pre-trained weights for the first run

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # Prevent overfitting
    layers.Dense(1, activation='sigmoid') # Binary Output (Edible vs Poisonous)
])

# 2. COMPILE MODEL
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. START TRAINING
print("\n--- [INFO] Starting Baseline Model Training (MobileNetV3) ---")
# Training for fewer epochs just to establish a baseline score
history = model.fit(
    train_generator,
    epochs=10, 
    validation_data=test_generator
)

# 4. SAVE BASELINE MODEL
# Saved as 'baseline' to distinguish from the final best Xception model
model.save('baseline_model.keras')
print("\n--- [SUCCESS] Baseline model saved as 'baseline_model.keras' ---")