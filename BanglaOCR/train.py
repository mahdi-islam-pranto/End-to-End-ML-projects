import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout,
    Dense, BatchNormalization, Reshape,
    Bidirectional, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Optional: force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR/"BanglaLekha-Isolated"/"Images_Sorborno"

batch_size = 32
img_size = (64, 64)
seed = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=seed
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=seed
)

num_classes = train_ds.cardinality().numpy()  # just to keep code consistent
print("Classes:", train_ds.class_names)
print("Train batches:", train_ds.cardinality().numpy())
print("Validation batches:", val_ds.cardinality().numpy())

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# Apply normalization to the datasets
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# Build The Model
input_shape = (64, 64, 1)

inputs = Input(shape=input_shape)

x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.30)(x)

x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.40)(x)

# 64 -> 32 -> 16 -> 8 after three 2x2 pools
x = Reshape((8, 8 * 128))(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Bidirectional(LSTM(128))(x)

x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

outputs = Dense(train_ds.element_spec[1].shape[-1], activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()