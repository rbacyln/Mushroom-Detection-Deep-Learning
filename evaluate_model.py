print("--- Libraries loading, please wait... ---")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# Import test data from our standardized data_loader
from data_loader import test_generator

# 1. LOAD THE TRAINED MODEL
print("--- [INFO] Loading Model and Test Data ---")
# CRITICAL: Using the new English filename
model = tf.keras.models.load_model('best_mushroom_model.keras')

# 2. RUN PREDICTIONS
# Ensuring the generator is reset to the beginning
test_generator.reset()

print("--- [INFO] Making Predictions on Test Set ---")
Y_pred = model.predict(test_generator)
y_pred = (Y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes

# 3. COMPUTE CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
target_names = ['Edible', 'Poisonous']

# 4. VISUALIZATION (HEATMAP)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Mushroom Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save the matrix image with an English filename
plt.savefig('confusion_matrix_english.png') 
print("\n--- [SUCCESS] Confusion Matrix saved as 'confusion_matrix_english.png' ---")

# 5. PRINT CLASSIFICATION REPORT
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=target_names))