# predict using the trained model pkl file
import pickle
import tensorflow as tf
import os
from pathlib import Path


# base directory
BASE_DIR = Path(__file__).resolve().parent
# load the model from the pickle file
with open(BASE_DIR / "bangla_ocr_model.pkl", "rb") as f:
    model = pickle.load(f)
    
    print("Model loaded from bangla_ocr_model.pkl")
    
# get some random samples from the validation dataset and predict
# Optional: force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR/"BanglaLekha-Isolated"/"Images_Sorborno"

# batch_size = 32
batch_size = 16
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



# check model accuracy on test set
loss, acc = model.evaluate(val_ds)
print("Validation Accuracy:", acc * 100)

# predict on a batch of validation data
for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    predicted_labels = tf.argmax(predictions, axis=1)
    true_labels = tf.argmax(labels, axis=1)
    
    print("Predicted labels:", predicted_labels.numpy())
    print("True labels:", true_labels.numpy())

