import ssl
# Bypass SSL verification for downloading ImageNet weights (if needed)
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models, callbacks
# Import data generators from our newly renamed data_loader.py
from data_loader import train_generator, test_generator

# 1. MODEL ARCHITECTURE: Xception (Transfer Learning)
# Xception is chosen for its depth and ability to capture fine texture details of mushrooms.
base_model = Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Fine-Tuning Strategy:
# Unfreeze the last 40 layers to adapt the model to our specific mushroom dataset
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Construct the final model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),       # Stabilize training
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),               # 50% Dropout to prevent overfitting
    layers.Dense(1, activation='sigmoid') # Binary classification (Edible vs Poisonous)
])

# 2. COMPILE MODEL
# Using a lower learning rate (1e-4) for stable fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. CALLBACKS (Smart Training)
# EarlyStopping: Stop training if validation accuracy doesn't improve for 5 epochs
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    restore_best_weights=True
)

# ModelCheckpoint: Save only the best performing model
checkpoint = callbacks.ModelCheckpoint(
    'best_mushroom_model.keras',  # Renamed to English
    save_best_only=True,
    monitor='val_accuracy'
)

print("\n--- [INFO] STARTING TRAINING (Target: 80%+ Accuracy) ---")
history = model.fit(
    train_generator,
    epochs=30,  # Extended training time
    validation_data=test_generator,
    callbacks=[early_stop, checkpoint]
)