# model_comparison.py
import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import *

# Set random seed for reproducibility
np.random.seed(42)

def create_model():
    """Create the model architecture"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # Get parent directory
validation_dir = os.path.join(current_dir, 'dataset', 'validation')
old_model_path = os.path.join(root_dir, 'wound_model_processed.h5')
new_model_path = os.path.join(root_dir, 'wound_model.h5')

# Print paths for verification
print(f"Validation data directory: {validation_dir}")
print(f"Old model path: {old_model_path}")
print(f"New model path: {new_model_path}")

# Image parameters
img_width, img_height = 224, 224
batch_size = 32

# Create data generator for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for maintaining order
)

# Load both models
print("\nLoading models...")
try:
    # Create new model instances
    old_model = create_model()
    new_model = create_model()
    
    # Load weights
    old_model.load_weights(old_model_path)
    new_model.load_weights(new_model_path)
    
    # Compile models
    old_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    new_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Get predictions from both models
print("\nGenerating predictions...")
old_predictions = old_model.predict(validation_generator)
validation_generator.reset()  # Reset generator for new predictions
new_predictions = new_model.predict(validation_generator)

# Get true labels
true_labels = validation_generator.classes
class_names = list(validation_generator.class_indices.keys())

# Convert predictions to class labels
old_pred_classes = np.argmax(old_predictions, axis=1)
new_pred_classes = np.argmax(new_predictions, axis=1)

# Generate classification reports
print("\nOld Model Performance:")
print(classification_report(true_labels, old_pred_classes, target_names=class_names))

print("\nNew Model Performance:")
print(classification_report(true_labels, new_pred_classes, target_names=class_names))

# Create confusion matrices
def plot_confusion_matrix(true_labels, predictions, title, class_names):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return cm

# Plot confusion matrices
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
old_cm = plot_confusion_matrix(true_labels, old_pred_classes, 
                             'Old Model Confusion Matrix', class_names)

plt.subplot(1, 2, 2)
new_cm = plot_confusion_matrix(true_labels, new_pred_classes, 
                             'New Model Confusion Matrix', class_names)

plt.savefig(os.path.join(current_dir, 'model_comparison.png'))
plt.close()

# Calculate per-class accuracy
def calculate_class_metrics(cm, class_names):
    metrics = {}
    for i, class_name in enumerate(class_names):
        true_positive = cm[i, i]
        false_negative = np.sum(cm[i, :]) - true_positive
        false_positive = np.sum(cm[:, i]) - true_positive
        true_negative = np.sum(cm) - true_positive - false_negative - false_positive
        
        accuracy = (true_positive + true_negative) / np.sum(cm)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    return metrics

# Calculate and compare metrics
old_metrics = calculate_class_metrics(old_cm, class_names)
new_metrics = calculate_class_metrics(new_cm, class_names)

# Print detailed comparison
print("\nDetailed Comparison by Wound Type:")
print("-" * 80)
print(f"{'Wound Type':<20} {'Metric':<10} {'Old Model':>10} {'New Model':>10} {'Difference':>10}")
print("-" * 80)

for wound_type in class_names:
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        old_value = old_metrics[wound_type][metric]
        new_value = new_metrics[wound_type][metric]
        difference = new_value - old_value
        print(f"{wound_type:<20} {metric:<10} {old_value:>10.3f} {new_value:>10.3f} {difference:>10.3f}")
    print("-" * 80)

# Save comparison results to file
results_file = os.path.join(current_dir, 'model_comparison_results.txt')
with open(results_file, 'w') as f:
    f.write("Model Comparison Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Old Model Performance:\n")
    f.write(classification_report(true_labels, old_pred_classes, target_names=class_names))
    f.write("\n\nNew Model Performance:\n")
    f.write(classification_report(true_labels, new_pred_classes, target_names=class_names))
    
    f.write("\n\nDetailed Comparison by Wound Type:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Wound Type':<20} {'Metric':<10} {'Old Model':>10} {'New Model':>10} {'Difference':>10}\n")
    f.write("-" * 80 + "\n")
    
    for wound_type in class_names:
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            old_value = old_metrics[wound_type][metric]
            new_value = new_metrics[wound_type][metric]
            difference = new_value - old_value
            f.write(f"{wound_type:<20} {metric:<10} {old_value:>10.3f} {new_value:>10.3f} {difference:>10.3f}\n")
        f.write("-" * 80 + "\n")

print("\nComparison analysis completed!")
print(f"- Confusion matrices saved as 'model_comparison.png'")
print(f"- Detailed results saved as 'model_comparison_results.txt'") 