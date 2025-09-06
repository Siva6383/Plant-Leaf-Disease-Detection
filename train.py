import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ========================
# SETTINGS
# ========================
DATA_DIR = "data"   # path to data folder
IMG_SIZE = 224      # resize images to 224x224
BATCH_SIZE = 32
EPOCHS = 10         # you can increase if you have GPU

# ========================
# LOAD DATA
# ========================
train_ds = image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_ds = image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("✅ Classes found:", class_names)

# Save class names to a JSON file (for later use in prediction)
os.makedirs("models", exist_ok=True)
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ========================
# DATA AUGMENTATION
# ========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ========================
# BUILD MODEL (Transfer Learning)
# ========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base layers

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ========================
# TRAIN MODEL
# ========================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/best_model.keras",
                                       save_best_only=True,
                                       monitor="val_accuracy",
                                       mode="max"),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                     restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ========================
# EVALUATE MODEL
# ========================
loss, acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {acc:.4f}")

# Save final model
model.save("models/leaf_disease_model.keras")
print("✅ Model saved to models/leaf_disease_model.keras")
