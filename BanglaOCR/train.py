import os
import json
from pathlib import Path
import pickle

import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout,
    Dense, BatchNormalization, Reshape,
    Bidirectional, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Optional: force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "BanglaLekha-Isolated" / "Images_Sorborno"

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE  = 16
IMG_SIZE    = (64, 64)
SEED        = 42
# EPOCHS      = 20          # bump up; early stopping will halt when needed
EPOCHS      = 15           # for quick testing; increase for real training
LEARNING_RATE = 1e-3
DROPOUT_1   = 0.25
DROPOUT_2   = 0.30
DROPOUT_3   = 0.40
DROPOUT_FC  = 0.50

# ── Load datasets ────────────────────────────────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

# ── Class names ──────────────────────────────────────────────────────────────
class_names = train_ds.class_names
num_classes = len(class_names)          # ✅ FIX: was train_ds.cardinality()

print(f"Classes  : {num_classes}")
print(f"Names    : {class_names}")
print(f"Train batches : {train_ds.cardinality().numpy()}")
print(f"Val   batches : {val_ds.cardinality().numpy()}")

# ── Save labels.json ─────────────────────────────────────────────────────────
# {index: class_name}  e.g. {"0": "ka", "1": "kha", ...}

print("Class mapping (index → folder name):")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Save the exact mapping the model was trained with
label_map = {str(i): name for i, name in enumerate(class_names)}
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("Saved labels.json ✓")

# ── Normalize ────────────────────────────────────────────────────────────────
rescale = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds   = val_ds.map(lambda x, y: (rescale(x), y))

# ── Build model ──────────────────────────────────────────────────────────────
def build_model(num_classes, learning_rate=LEARNING_RATE):
    inputs = Input(shape=(64, 64, 1))

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(DROPOUT_1)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(DROPOUT_2)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(DROPOUT_3)(x)

    # 64 → 32 → 16 → 8 after three 2×2 pools
    x = Reshape((8, 8 * 128))(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128))(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_FC)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model(num_classes)
model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
checkpoint_cb = ModelCheckpoint(
    filepath="bangla_ocr_model.keras",     # ✅ saves best model with the required filename
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

early_stop_cb = EarlyStopping(
    monitor="val_accuracy",
    patience=5,                 # stop if no improvement for 5 epochs
    restore_best_weights=True,
    verbose=1,
)

# ── MLflow experiment ────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("BanglaLekha-OCR")

with mlflow.start_run():

    # 1. Log hyperparameters
    mlflow.log_params({
        "batch_size":    BATCH_SIZE,
        "img_size":      IMG_SIZE,
        "epochs":        EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dropout_conv1": DROPOUT_1,
        "dropout_conv2": DROPOUT_2,
        "dropout_conv3": DROPOUT_3,
        "dropout_fc":    DROPOUT_FC,
        "num_classes":   num_classes,
        "optimizer":     "Adam",
        "seed":          SEED,
    })
    
    # log model summary for testing the mlflow
    model_summary_str = []
    model.summary(print_fn=lambda x: model_summary_str.append(x))
    mlflow.log_text("\n".join(model_summary_str), "model_summary.txt")

    # 2. Train
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        # callbacks=[checkpoint_cb, early_stop_cb],
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=1,
    )

    # 3. Log per-epoch metrics
    for epoch, (t_loss, t_acc, v_loss, v_acc) in enumerate(zip(
        history.history["loss"],
        history.history["accuracy"],
        history.history["val_loss"],
        history.history["val_accuracy"],
    )):
        mlflow.log_metrics(
            {
                "train_loss":    t_loss,
                "train_accuracy": t_acc,
                "val_loss":      v_loss,
                "val_accuracy":  v_acc,
            },
            step=epoch,
        )

    # 4. Final evaluation on validation set
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    print(f"\nFinal Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"Final Validation Loss     : {val_loss:.4f}")

    mlflow.log_metrics({
        "final_val_loss":     val_loss,
        "final_val_accuracy": val_acc,
    })

    # save the pkl model 
    with open("bangla_ocr_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved bangla ocr model.pkl ✓")
    mlflow.log_artifact("bangla_ocr_model.pkl")

    # 5. Log saved artifacts
    mlflow.log_artifact("bangla_ocr_model.keras")   # best model
    mlflow.log_artifact("bangla_ocr_model.pkl")
    # mlflow.log_artifact("labels.json")   # class-label mapping

    # 6. Log the Keras model in MLflow's native format (enables mlflow.pyfunc.load_model)
    mlflow.keras.log_model(model, artifact_path="keras_model")

    print("\nMLflow run complete. Artifacts logged ✓")

print("Training finished.")
print("  model.keras  → best checkpoint saved")
print("  labels.json  → class-label mapping saved")