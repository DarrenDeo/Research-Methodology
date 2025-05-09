**News Fact Check Web Application**

**Table of Contents**

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)

   * [Running the Application](#running-the-application)
   * [API Endpoint](#api-endpoint)
6. [Model Details](#model-details)

   * [Traditional Models](#traditional-models)
   * [Transformer Models](#transformer-models)
   * [Model Performance](#model-performance)
7. [Retraining and Extending](#retraining-and-extending)
8. [Requirements](#requirements)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview

This web application allows users to input a news headline or URL and receive a real-time classification indicating whether the news is likely a hoax or not. The application also categorizes the news into topics such as politics, economy, and education before running the hoax detection model.

## Directory Structure

```
News Fact Check/
├── app.py                   # Main Flask application
├── model.py                 # Data download, preprocessing, training, and inference functions
├── models/                  # Serialized trained models
│   ├── traditional_model_all.pkl
│   ├── traditional_model_economy.pkl
│   ├── traditional_model_education.pkl
│   └── traditional_model_politics.pkl
├── model_performance_comparison.png  # Visualization comparing model results
├── model_performance_results.csv     # Performance metrics (root-level)
├── static/                  # Frontend assets and duplicates of performance files
│   ├── index.html
│   └── model_performance_results.csv
├── requirements.txt         # Python dependencies
└── __pycache__/             # Compiled Python cache files
```

## Features

* **URL or Headline Input:** Users can submit either a URL or a raw headline.
* **Category Detection:** Classifies news into topics (politics, economy, education, etc.).
* **Hoax Detection:** Predicts whether the news is a hoax or genuine.
* **Confidence Score:** Returns a confidence value indicating model certainty.
* **Interactive Frontend:** Simple HTML interface served via Flask.

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd "News Fact Check"
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Start the Flask server:

```bash
python app.py
```

By default, the app runs on **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**. Open this URL in your browser to access the frontend.

### API Endpoint

* **POST** `/api/check-news`

  * **Request Body (JSON):**

    ```json
    {
      "urlOrHeadline": "<news_url_or_headline>"
    }
    ```
  * **Response (JSON):**

    ```json
    {
      "headline": "<resolved_headline>",
      "category": "<detected_category>",
      "prediction": "Hoax" | "Not Hoax",
      "confidence": <confidence_score>
    }
    ```

## Model Details

### Traditional Models

* **TF-IDF + Multinomial Naive Bayes** pipelines trained on category-specific datasets.
* Stored in `models/traditional_model_*.pkl` files.

### Transformer Models

* Placeholder for future integration of BERT/XLM-RoBERTa-based classifiers (e.g., `AutoModelForSequenceClassification`).
* Tokenizers and training scripts defined in `model.py`.

### Model Performance

* **CSV Results:** `model_performance_results.csv` contains metrics such as accuracy, F1-score per model and category.
* **Performance Plot:** `model_performance_comparison.png` visualizes comparative performance across models.

## Retraining and Extending

1. **Download Datasets**

   * Uses `kagglehub` to fetch datasets for each category (politics, economy, education, religion).
   * Fallback options provided in `model.py`.

2. **Preprocessing & Training**

   * Functions in `model.py` handle tokenization, TF-IDF vectorization, train-test split, and model serialization.

3. **Add New Categories**

   * Update dataset download links in `download_datasets()`.
   * Retrain models by calling the appropriate functions in `model.py`.

## Requirements

* Python 3.8+
* Flask
* Flask-CORS
* scikit-learn
* transformers
* torch
* pandas
* numpy
* nltk
* beautifulsoup4
* requests
* kagglehub

Install all dependencies with:

```bash
pip install -r requirements.txt
```

then run

```bash
python app.py
```
