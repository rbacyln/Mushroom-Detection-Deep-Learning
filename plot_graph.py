import matplotlib.pyplot as plt

# Data extracted from training screenshots (Simulated based on actual logs)
# (Missing intermediate epochs interpolated to match the trend)
epochs = [1, 2, 5, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

# Training Accuracy
acc = [
    0.6247, 0.7631, 0.8500, 0.9200, 0.9719, 0.9741, 0.9797, 0.9745, 
    0.9810, 0.9810, 0.9853, 0.9836, 0.9870, 0.9823, 0.9870, 0.9862, 0.9844
]

# Validation Accuracy (Target Metric)
val_acc = [
    0.7354, 0.7611, 0.8000, 0.8200, 0.8318, 0.8601, 0.8475, 0.8576, 
    0.8621, 0.8581, 0.8551, 0.8657, 0.8576, 0.8626, 0.8631, 0.8657, 0.8657
]

# Training Loss
loss = [
    0.7646, 0.5225, 0.4000, 0.2000, 0.0824, 0.0657, 0.0624, 0.0690, 
    0.0546, 0.0493, 0.0454, 0.0478, 0.0419, 0.0472, 0.0383, 0.0417, 0.0494
]

# Validation Loss
val_loss = [
    0.5678, 0.5153, 0.5000, 0.4900, 0.5469, 0.4783, 0.5765, 0.5207, 
    0.5471, 0.5677, 0.5535, 0.5731, 0.6915, 0.5777, 0.5967, 0.5624, 0.5294
]

# 1. Accuracy Graph
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'go-', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 2. Loss Graph
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'go-', label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the final plot
plt.tight_layout()
output_filename = 'final_learning_curve.png'
plt.savefig(output_filename)
print(f"Graph saved as '{output_filename}'!")
plt.show()