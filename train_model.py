import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import json

# -----------------------------
# Paths and Config
# -----------------------------
train_dir = os.path.join("dataset", "train")
valid_dir = os.path.join("dataset", "valid")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Slightly larger batch for stability

# -----------------------------
# Enhanced Data Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

# Save class names for the Flask app
class_names = list(train_data.class_indices.keys())
os.makedirs("model", exist_ok=True)
with open("model/classes.json", "w") as f:
    json.dump(class_names, f)

# -----------------------------
# Model Architecture
# -----------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Start with base frozen

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(len(class_names), activation="softmax")
])

# -----------------------------
# Training Strategy: Phase 1 (Warmup)
# -----------------------------
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

print("Phase 1: Training top layers...")
history_warmup = model.fit(train_data, validation_data=valid_data, epochs=5)

# -----------------------------
# Training Strategy: Phase 2 (Fine-Tuning)
# -----------------------------
print("Phase 2: Fine-tuning base model...")
base_model.trainable = True
# Freeze early layers, unfreeze late layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Lower learning rate is CRITICAL for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
]

history_fine = model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=25, 
    callbacks=callbacks
)

# -----------------------------
# Visualization (Accuracy & Loss Curves)
# -----------------------------
def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('model/training_curves.png')
    plt.show()

plot_history(history_warmup, history_fine)

model.save("model/skin_model.keras")
print("Training complete! Curves saved to model/training_curves.png")
