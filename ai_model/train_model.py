import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

# -----------------------------
# 1. Parse input arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", required=True, help="Path to training dataset")
parser.add_argument("--val_dir", required=True, help="Path to validation dataset")
args = parser.parse_args()

# -----------------------------
# 2. Image Data Generators
# -----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Data augmentation including wave-shift like effect
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,  # wave-like horizontal shift
    height_shift_range=0.1, # wave-like vertical shift
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    args.val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -----------------------------
# 3. Build CNN model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# -----------------------------
# 4. Train CNN
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# -----------------------------
# 5. Save model and classes
# -----------------------------
model.save("model.h5")

# Save class names
with open("classes.txt", "w") as f:
    f.write("\n".join(list(train_generator.class_indices.keys())))

print("✅ Training complete! Model saved as model.h5 and classes.txt")
