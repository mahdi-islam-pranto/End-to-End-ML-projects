# BanglaOCR — Handwritten Bangla Character OCR

This repository contains a lightweight OCR for isolated Bangla characters and a Streamlit front-end to draw words, segment them into characters, and predict each character using a CNN+BiLSTM model.

## Project Structure

- `app.py` — Streamlit UI and segmentation code
- `train.py` — training script (uses TensorFlow / Keras and MLflow)
- `labels.json` — index → label mapping used by the app
- `bangla_ocr_model.keras` — best Keras checkpoint (artifact)
- `bangla_ocr_model.pkl` — optional pickled model
- `dockerfile` — Dockerfile for building a containerized Streamlit app
- `requirements.txt` — Python dependencies
- `BanglaLekha-Isolated/` — dataset (not included here in full)

## Overview / Approach

1. Dataset: The model is trained on the BanglaLekha-Isolated dataset (project uses the `Images` / `Images_Sorborno` folders). The training script expects data organized as one folder per class (standard ImageNet-style layout).
2. Preprocessing: Images are loaded in grayscale, resized to `64×64`, and rescaled to `[0,1]` via a `Rescaling(1/255)` layer in the pipeline.
3. Model architecture: a small CNN feature extractor followed by BiLSTM and a dense classifier. Key parts:
   - Conv blocks: multiple Conv2D + BatchNormalization + MaxPooling + Dropout
   - Reshape to a sequence and pass through `Bidirectional(LSTM(128))` layers
   - Dense(256) → BatchNormalization → Dropout → Dense(num_classes, softmax)
4. Training: `train.py` compiles with `Adam(1e-3)` and trains using `categorical_crossentropy` with accuracy metric. Best checkpoint saved to `bangla_ocr_model.keras` and a `labels.json` mapping is saved.
5. MLflow: training runs are logged to an MLflow server. The script sets `mlflow.set_tracking_uri("http://127.0.0.1:8080")` and `mlflow.set_experiment("BanglaLekha-OCR")`. The script logs parameters, per-epoch metrics, model summary (`model_summary.txt`), the Keras model via `mlflow.keras.log_model`, and artifact files.

## Dataset Preparation

- Acquire the BanglaLekha-Isolated dataset and arrange images into class folders.
- The training script uses `tf.keras.utils.image_dataset_from_directory` with `color_mode="grayscale"`, `image_size=(64,64)`, and a `validation_split=0.2`.
- During training, `labels.json` is created automatically by `train.py` to map numeric indices (used by the model) to readable class labels.

## Preprocessing Details

- Images normalized to `[0,1]` by `Rescaling(1.0/255)`.
- Model expects shape `(64, 64, 1)` (grayscale).
- Streamlit frontend performs canvas → RGBA → grayscale conversion, Otsu binarization, and per-character preprocessing including inversion and resizing to `64×64`.

## Word Segmentation Strategy (used in `app.py`)

The Streamlit app segments a hand-drawn word (canvas image) into character crops using this pipeline:

1. Binarize the grayscale canvas using Otsu thresholding so strokes become white on black (or inverted when needed).
2. Find connected components with `cv2.connectedComponentsWithStats` to get candidate bounding boxes.
3. Filter small/noise components and collect bounding boxes.
4. Merge horizontally overlapping/nearby boxes (to keep matra and diacritics attached to their base glyph). A horizontal gap threshold (`H_GAP`) is computed relative to image width.
5. Sort merged boxes left-to-right and extract crops with a small padding (e.g., `PAD=6`).
6. Each crop is inverted (to match training data where strokes are dark on light background), resized to `64×64`, normalized, and passed through the model.

This approach works well for isolated handwritten glyphs and short words where characters are not heavily touching. It intentionally groups diacritics with base glyphs so predictions correspond to base+sign combinations present in the training labels.

## How to Train

1. Create and activate a Python virtual environment (recommended):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
source .venv/bin/activate        # macOS / Linux
```

2. Install dependencies:

```
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run MLflow UI (optional) to monitor runs locally on `http://127.0.0.1:8080`:

```
mlflow ui --port 8080
```

4. Start training:

```
python train.py
```

5. Artifacts saved by the script:

- `bangla_ocr_model.keras` (best checkpoint)
- `bangla_ocr_model.pkl` (pickled model)
- `labels.json` (index → label mapping)
- `model_summary.txt` logged to MLflow

Note: `train.py` currently runs on CPU by default (`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`). Remove or modify that line to enable GPU usage if available.

## Running the Streamlit App (local)

1. Ensure `bangla_ocr_model.keras` and `labels.json` are in the same folder as `app.py` (they are copied into container in the `dockerfile`).
2. Run locally:

```
streamlit run app.py --server.port=8502
```

3. Open `http://localhost:8502` in your browser and draw words on the canvas. Use the `Predict word` button to segment and predict characters. The UI shows per-character confidence and a character-level breakdown.

## Docker (build & run)

Build the image (example from `BanglaOCR` folder):

```
docker build -t bangla-ocr-streamlit:0.1 .
```

Run the container mapping port `8502`:

```
docker run -p 8502:8502 --rm bangla-ocr-streamlit:0.1
```

Notes:

- Dockerfile copies `labels.json`, `app.py`, `bangla_ocr_model.keras`, and `bangla_ocr_model.pkl` into the image. Ensure these artifacts exist prior to building the image.
- Healthcheck is configured to hit the Streamlit health endpoint `/_stcore/health`.

## MLflow Integration

- `train.py` logs parameters, per-epoch metrics, final metrics, artifacts, and the Keras model. Example tracking URI used by the script:

```
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("BanglaLekha-OCR")
```

- To view runs, start MLflow UI (`mlflow ui --port 8080`) and visit `http://127.0.0.1:8080`.

## Limitations

- Model trained on isolated characters (and some character+diacritic combos). It is not a full word-level seq2seq OCR engine. It classifies single glyphs cropped from the canvas.
- Conjunct consonants and connected/cursive handwriting may be segmented incorrectly or produce wrong labels if they were not present in training data.
- Matra and diacritics can sometimes be separated from the base character depending on handwriting; the current merging heuristics try to mitigate this but are not flawless.
- The model in the repo is CPU-trained; for larger datasets / better accuracy, training on GPU is recommended.

## Possible Improvements

- Increase dataset size and include more handwritten variations (writers, pens, sizes).
- Use stronger augmentation (elastic transforms, random erode/dilate, contrast jitter) to improve robustness.
- Replace fixed-class classifier with a sequence model trained with CTC or encoder-decoder if you want whole-word transcription without segmentation.
- Improve segmentation: use learned instance segmentation (U-Net / Mask R-CNN) or an RNN/Transformer-based layout parser to handle touching characters and conjuncts.
- Quantize or convert the model (TFLite, ONNX) for faster inference on edge devices.
- Add language model rescoring to improve word-level coherence.

## Where to look in the repo

- Model training: [train.py](train.py)
- Streamlit frontend & segmentation: [app.py](app.py)
- Docker: [dockerfile](dockerfile)
- Labels mapping: [labels.json](labels.json)

## Quick References / Commands

- Train locally: `python train.py`
- Run Streamlit app locally: `streamlit run app.py --server.port=8502`
- Run MLflow UI: `mlflow ui --port 8080`
- Build Docker image: `docker build -t bangla-ocr-streamlit:0.1 .`
- Run Docker container: `docker run -p 8502:8502 --rm bangla-ocr-streamlit:0.1`

---


## Screenshots

Below are screenshots from the project (Streamlit UI, MLflow tracking and metrics, Docker build view).

### Streamlit app (word recognizer)

![Streamlit app drawing canvas](screenshots/বাংলা-OCR%20streamlit%20docker.png)

### MLflow run overview

![MLflow run overview](screenshots/ml%20flow%20tracking.png)

### MLflow metrics & artifacts

![MLflow metrics and artifacts](screenshots/mlflow%20traking%20matrics.png)

### MLflow artifacts view

![MLflow artifacts view](screenshots/mlflow%20tracking%20artifacts.png)
