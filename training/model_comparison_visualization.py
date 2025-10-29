import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

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
root_dir = os.path.dirname(current_dir)
validation_dir = os.path.join(current_dir, 'dataset', 'validation')
old_model_path = os.path.join(root_dir, 'wound_model_processed.h5')
new_model_path = os.path.join(root_dir, 'wound_model.h5')

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
    shuffle=False
)

# Load both models
print("Loading models...")
old_model = create_model()
new_model = create_model()

old_model.load_weights(old_model_path)
new_model.load_weights(new_model_path)

old_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Get predictions
print("Generating predictions...")
old_predictions = old_model.predict(validation_generator)
validation_generator.reset()
new_predictions = new_model.predict(validation_generator)

# Get true labels and class names
true_labels = validation_generator.classes
class_names = list(validation_generator.class_indices.keys())

# Convert predictions to class labels
old_pred_classes = np.argmax(old_predictions, axis=1)
new_pred_classes = np.argmax(new_predictions, axis=1)

# Generate classification reports
old_report = classification_report(true_labels, old_pred_classes, target_names=class_names, output_dict=True)
new_report = classification_report(true_labels, new_pred_classes, target_names=class_names, output_dict=True)

# Create DataFrame for comparison
comparison_data = []
for class_name in class_names:
    row = {
        'Class': class_name,
        'Old Precision': old_report[class_name]['precision'],
        'New Precision': new_report[class_name]['precision'],
        'Old Recall': old_report[class_name]['recall'],
        'New Recall': new_report[class_name]['recall'],
        'Old F1': old_report[class_name]['f1-score'],
        'New F1': new_report[class_name]['f1-score'],
        'Support': old_report[class_name]['support']
    }
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)

# Add overall metrics
overall_row = {
    'Class': 'Overall',
    'Old Precision': old_report['weighted avg']['precision'],
    'New Precision': new_report['weighted avg']['precision'],
    'Old Recall': old_report['weighted avg']['recall'],
    'New Recall': new_report['weighted avg']['recall'],
    'Old F1': old_report['weighted avg']['f1-score'],
    'New F1': new_report['weighted avg']['f1-score'],
    'Support': old_report['weighted avg']['support']
}
df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

# Save to CSV
df.to_csv('model_comparison_table.csv', index=False)

# Create visualizations
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(20, 15))

# 1. Bar plot comparing F1 scores
plt.subplot(2, 2, 1)
x = np.arange(len(class_names))
width = 0.35
plt.bar(x - width/2, df['Old F1'][:-1], width, label='Old Model', color='skyblue')
plt.bar(x + width/2, df['New F1'][:-1], width, label='New Model', color='lightcoral')
plt.xlabel('Wound Type')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison by Wound Type')
plt.xticks(x, class_names, rotation=45, ha='right')
plt.legend()

# 2. Precision-Recall comparison
plt.subplot(2, 2, 2)
for i, class_name in enumerate(class_names):
    plt.scatter(df['Old Precision'][i], df['Old Recall'][i], label=f'Old {class_name}', alpha=0.6)
    plt.scatter(df['New Precision'][i], df['New Recall'][i], label=f'New {class_name}', alpha=0.6)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. Support distribution
plt.subplot(2, 2, 3)
plt.bar(class_names, df['Support'][:-1], color='lightgreen')
plt.xlabel('Wound Type')
plt.ylabel('Number of Samples')
plt.title('Distribution of Samples by Wound Type')
plt.xticks(rotation=45, ha='right')

# 4. Improvement in metrics
plt.subplot(2, 2, 4)
improvements = {
    'Precision': df['New Precision'] - df['Old Precision'],
    'Recall': df['New Recall'] - df['Old Recall'],
    'F1': df['New F1'] - df['Old F1']
}
x = np.arange(len(class_names))
width = 0.25
plt.bar(x - width, improvements['Precision'][:-1], width, label='Precision', color='lightblue')
plt.bar(x, improvements['Recall'][:-1], width, label='Recall', color='lightgreen')
plt.bar(x + width, improvements['F1'][:-1], width, label='F1', color='lightcoral')
plt.xlabel('Wound Type')
plt.ylabel('Improvement')
plt.title('Improvement in Metrics (New - Old)')
plt.xticks(x, class_names, rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison_visualization.png', bbox_inches='tight', dpi=300)
plt.close()

# Print the comparison table
print("\nModel Comparison Table:")
print(df.to_string(index=False))

print("\nVisualizations have been saved as 'model_comparison_visualization.png'")
print("Detailed comparison table has been saved as 'model_comparison_table.csv'") 