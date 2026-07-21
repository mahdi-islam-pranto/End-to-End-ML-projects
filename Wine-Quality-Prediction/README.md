# End-to-End Machine Learning Project with MLflow

This project demonstrates a complete machine learning workflow for predicting wine quality using a regression model. It covers everything from data ingestion and validation to model training, evaluation, and deployment through a simple Flask web application.

## Project Overview

The application uses the red wine quality dataset and trains a machine learning model to predict quality scores based on physicochemical features such as acidity, sugar, pH, alcohol, and sulphates.

### Key Features

- End-to-end ML pipeline with modular components
- Data validation and transformation steps
- Model training and evaluation with MLflow tracking
- Flask-based web interface for predictions
- Docker support for easy deployment

## Tech Stack

- Python 3.8+
- Flask
- scikit-learn
- pandas, numpy, matplotlib
- MLflow
- joblib, PyYAML, python-box
- Docker

## Project Structure

- app.py: Flask web app entry point
- main.py: orchestrates the full training pipeline
- src/mlProject/: package containing pipeline, components, config, and utilities
- templates/: HTML pages for the web UI
- static/: CSS and JavaScript assets
- artifacts/: generated data, model, and evaluation outputs
- research/: Jupyter notebooks for experimentation

## Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd End-to-end-Machine-Learning-Project-with-MLflow
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Running the Training Pipeline

To run the complete training workflow:

```bash
python main.py
```

This executes the following stages:

- Data ingestion
- Data validation
- Data transformation
- Model training
- Model evaluation

## Running the Web Application

Start the Flask app:

```bash
python app.py
```

Then open your browser at:

```text
http://localhost:8080
```

### Available Routes

- GET /: Home page
- GET /train: Trigger the training pipeline
- POST /predict: Submit feature values and receive a prediction

## MLflow Tracking

The project uses MLflow to log experiment metadata and model-related artifacts. You can launch the MLflow UI locally with:

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

## Docker

Build the Docker image:

```bash
docker build -t wine-quality-app .
```

Run the container:

```bash
docker run -p 8080:8080 wine-quality-app
```

## Output Artifacts

After training, the following files are generated in the artifacts folder:

- Data files for train/test splits
- Validation status report
- Trained model file
- Evaluation metrics

## License

This project is licensed under the MIT License.
