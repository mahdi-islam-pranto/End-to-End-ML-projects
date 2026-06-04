import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "BanglaLekha-Isolated" / "Images_Sorborno"

ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(64, 64),
)

print("Actual class order assigned by the dataset:")
for i, name in enumerate(ds.class_names):
    print(f"  {i}: '{name}'")
    

ds = tf.keras.utils.image_dataset_from_directory(
    "BanglaLekha-Isolated/Images_Sorborno",
    labels="inferred", label_mode="categorical",
    color_mode="grayscale", image_size=(64,64)
)
print(ds.class_names)  # prints the exact folder order