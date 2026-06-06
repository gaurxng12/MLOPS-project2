# 🧠 Sentiment Analysis MLOps Pipeline

> An end-to-end MLOps project that automates sentiment classification of user-generated text — from data ingestion and model training to experiment tracking, containerized deployment, and CI/CD automation.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Pipeline Stages](#-pipeline-stages)
- [Model Details](#-model-details)
- [API & Web Interface](#-api--web-interface)
- [Monitoring](#-monitoring)
- [CI/CD Workflow](#-cicd-workflow)
- [Getting Started](#-getting-started)
- [Running with Docker](#-running-with-docker)
- [Experiment Tracking](#-experiment-tracking)
- [Model Registry & Promotion](#-model-registry--promotion)
- [Configuration](#-configuration)
- [Results](#-results)

---

# 📌 Overview

This project builds a binary sentiment classification system capable of predicting whether a user review expresses a **positive** or **negative** sentiment.

Beyond model training, the project demonstrates a complete MLOps workflow including:

* Automated data pipelines using DVC
* Data and model versioning with AWS S3
* Experiment tracking using MLflow on DagsHub
* Flask-based model serving
* Containerization with Docker
* Automated testing and CI/CD using GitHub Actions

---

# 🎯 Problem Statement

User-generated content such as reviews, comments, and feedback contains valuable sentiment information that organizations can use to improve products and customer experience.

Manual sentiment analysis is costly and difficult to scale. This project automates the full machine learning lifecycle:

* Training a sentiment classifier
* Tracking experiments and metrics
* Versioning datasets and artifacts
* Serving predictions through a REST API
* Validating model quality through automated tests

The result is a reproducible and production-oriented machine learning workflow.

---

# ✨ Key Features

* End-to-end DVC pipeline
* Automated data preprocessing
* Bag-of-Words feature engineering
* Logistic Regression sentiment classifier
* Data and model versioning using DVC and AWS S3
* Experiment tracking with MLflow
* Flask REST API
* Dockerized deployment
* GitHub Actions CI/CD
* Automated model validation tests

---

# 🏗 Architecture

```text
GitHub Actions
        │
        ▼
    DVC Pipeline
        │
        ▼
Data Ingestion
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
MLflow (DagsHub)
        │
        ▼
Flask Application
```

Data and model artifacts are versioned using DVC with AWS S3 as the remote storage backend, enabling reproducible experiments and dataset tracking.

---

# 🛠 Tech Stack

| Category                | Technology                     |
| ----------------------- | ------------------------------ |
| Language                | Python 3.10                    |
| Machine Learning        | Scikit-Learn                   |
| NLP                     | NLTK                           |
| Feature Engineering     | CountVectorizer (Bag-of-Words) |
| Experiment Tracking     | MLflow                         |
| Remote Tracking         | DagsHub                        |
| Data & Model Versioning | DVC                            |
| Storage                 | AWS S3                         |
| Web Framework           | Flask                          |
| Containerization        | Docker                         |
| CI/CD                   | GitHub Actions                 |

---

# 📁 Project Structure

```text
MLOPS-project2/
├── src/
│   ├── data
│   ├── features
│   ├── model
│   └── connections
│
├── flask_app/
├── tests/
├── reports/
│
├── dvc.yaml
├── params.yaml
├── Dockerfile
└── .github/workflows/
```

---

# ⚙️ Pipeline Stages

The project uses a DVC pipeline consisting of:

1. Data Ingestion
2. Data Preprocessing
3. Feature Engineering (Bag-of-Words)
4. Model Training & Evaluation
5. Experiment Tracking with MLflow
6. Model Serving with Flask

Run the entire pipeline:

```bash
dvc repro
```

---

# 🤖 Model Details

| Component           | Details                                   |
| ------------------- | ----------------------------------------- |
| Algorithm           | Logistic Regression                       |
| Features            | Bag-of-Words (CountVectorizer)            |
| Text Processing     | Cleaning, Stopword Removal, Lemmatization |
| Experiment Tracking | MLflow                                    |
| Evaluation Metrics  | Accuracy, Precision, Recall, F1, AUC      |

All model parameters are configurable through `params.yaml`.

---

# 🌐 API & Web Interface

The Flask application loads the latest trained model and serves predictions through both a web interface and REST API.

### Web Interface

Navigate to:

```text
http://localhost:5000
```

and submit text for sentiment prediction.

### REST Endpoints

| Method | Endpoint   | Description                                        |
| ------ | ---------- | -------------------------------------------------- |
| GET    | `/`        | Renders the sentiment prediction UI                |
| POST   | `/predict` | Accepts text input and returns predicted sentiment |

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
-d "text=This product is absolutely fantastic!"
```

---

# 🔄 CI/CD Workflow

The GitHub Actions workflow automatically runs on every push.

```text
Push to GitHub
      │
      ▼
Install Dependencies
      │
      ▼
dvc repro --force
      │
      ▼
Model Tests
      │
      ▼
Flask API Tests
```

The workflow ensures that pipeline execution and application functionality are validated automatically.

---

# 🚀 Getting Started

## Prerequisites

* Python 3.10
* Conda or Virtual Environment
* AWS Account (for S3 storage)
* DagsHub Account

## Clone Repository

```bash
git clone https://github.com/bgaurangan/MLOPS-project2.git

cd MLOPS-project2
```

## Create Environment

```bash
conda create -n sentiment python=3.10

conda activate sentiment
```

## Install Dependencies

```bash
pip install -r flask_app/requirements.txt

pip install -e .
```

## Configure Credentials

Set your DagsHub token:

```bash
export CAPSTONE_TEST=<your-dagshub-token>
```

## Pull Versioned Data

```bash
dvc pull
```

## Run Pipeline

```bash
dvc repro
```

## Run Tests

```bash
python -m unittest tests/test_model.py

python -m unittest tests/test_flask_app.py
```

---

# 🐳 Running with Docker

Build the Docker image:

```bash
docker build -t sentiment-app .
```

Run the container:

```bash
docker run -p 5000:5000 \
-e CAPSTONE_TEST=<your-dagshub-token> \
sentiment-app
```

Application URL:

```text
http://localhost:5000
```

---

# 🔬 Experiment Tracking

Training runs are tracked using MLflow hosted on DagsHub.

Each run records:

* Hyperparameters
* Evaluation metrics
* Trained model artifacts
* Experiment metadata

This enables reproducibility, experiment comparison, and model performance tracking over time.

---

# 📈 Results

Current model performance on the held-out test set:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 75.3% |
| Precision | 80.4% |
| Recall    | 63.4% |
| AUC       | 84.7% |

Automated tests ensure that model quality remains above predefined thresholds before deployment.

---

# 📄 License

MIT License — see the LICENSE file for details.

---

### 👤 Author

**Gaurang Bhogle**

