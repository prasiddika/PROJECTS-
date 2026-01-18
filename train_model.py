# train_model.py

# 1️⃣ Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 2️⃣ Set paths to your dataset
train_dir = os.path.join("dataset", "train")
valid_dir = os.path.join("dataset", "valid")

# 3️⃣ Image parameters
IMG_SIZE = (224, 224)  # MobileNetV2 expects 224x224 images
BATCH_SIZE = 8      # number of images processed at once

# 4️⃣ Preprocessing for training and validation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)



valid_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# 5️⃣ Load images from folders
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"  # 4 classes → one-hot encoded
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# 6️⃣ Build the model using MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = True
for layer in base_model.layers[:-30]:  # keep first layers frozen
    layer.trainable = False


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # converts feature maps to a single vector
    Dense(128, activation="relu"),  # small dense layer to learn skin features
    Dense(4, activation="softmax")   # 4 skin types
])

# 7️⃣ Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 8️⃣ Train the model
EPOCHS = 12  # more epoch for better learning 
history = model.fit(train_data, validation_data=valid_data, epochs=EPOCHS)
print("Final training accuracy:", history.history["accuracy"][-1])
print("Final validation accuracy:", history.history["val_accuracy"][-1])


# 9️⃣ Save the trained model
os.makedirs("model", exist_ok=True)  # create model folder if it doesn't exist
model.save("model/skin_model.h5")
print("Training complete! Model saved as model/skin_model.h5")
