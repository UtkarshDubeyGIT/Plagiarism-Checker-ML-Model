# Plagiarism & AI Text Detection API

This repository provides a Flask-based API for detecting AI-generated text and checking for plagiarism between two text samples. It leverages machine learning models trained on TF-IDF features and XGBoost for robust detection.

## Features

- **AI Text Detection**: Classifies whether a given text is likely AI-generated.
- **Plagiarism Detection**: Compares two texts and predicts if one is plagiarized from the other, using advanced text similarity and ML features.
- **Pre-trained Models**: Includes pre-trained models and vectorizers for immediate use.

## Project Structure

```
ai_model.pkl                  # Trained model for AI text detection
ai_tfidf_vectorizer.pkl       # TF-IDF vectorizer for AI detection
app.py                        # Flask API application
xgb_tuned_model.pkl           # XGBoost model for plagiarism detection
xgb_tfidf_vectorizer.pkl      # TF-IDF vectorizer for plagiarism detection
xgb_tuned_plagiarism_model.ipynb # Notebook for XGBoost model training
dataset.csv                   # Dataset for training and evaluation
model_traing.ipynb            # Notebook for baseline model training
requirements.txt              # Python dependencies
test_my_models.py             # Script to test API endpoints
```

## API Endpoints

### 1. `/detect-ai` (POST)
- **Input**: JSON with a `text` field.
- **Output**: JSON with `prediction` (0=not AI, 1=AI) and `probability`.

**Example:**
```json
{
  "text": "Sample text to check."
}
```

### 2. `/check-plagiarism` (POST)
- **Input**: JSON with `source` and `suspicious` fields.
- **Output**: JSON with `prediction` (0=not plagiarized, 1=plagiarized) and `probability`.

**Example:**
```json
{
  "source": "Original text.",
  "suspicious": "Text to check for plagiarism."
}
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the API Server
```bash
python app.py
```

The server will start on `http://0.0.0.0:5000` by default.

### 3. Test the API
You can use the provided `test_my_models.py` script:
```bash
python test_my_models.py
```

Or use `curl`/Postman to send requests to the endpoints.

## Model Training
- See `model_traing.ipynb` for baseline models (Logistic Regression, Random Forest, SVM, Naive Bayes).
- See `xgb_tuned_plagiarism_model.ipynb` for advanced XGBoost-based plagiarism detection.

## Dataset
- The dataset (`dataset.csv`) contains pairs of source and plagiarized texts with labels (1=plagiarized, 0=not plagiarized).

## Notes
- All models and vectorizers are pre-trained and included in the repo.
- For retraining, refer to the Jupyter notebooks.

## License
This project is for educational and research purposes only.
