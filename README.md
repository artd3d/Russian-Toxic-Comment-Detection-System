# Russian Toxic Comment Detection System

A machine learning-based system for detecting toxic comments in Russian text using Natural Language Processing (NLP) and classification algorithms.

## ⚡ Quick Start

Get the API running in under 2 minutes:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and export the model
python model_exporter.py

# 3. Start the API server
python api.py

# 4. Test the API (in another terminal)
python test_api.py
```

**API will be available at:** `http://localhost:8000`  
**Interactive docs:** `http://localhost:8000/docs`

## 📋 Project Overview

This project implements a binary classification system to identify toxic comments in Russian language text. The system uses advanced NLP preprocessing techniques combined with TF-IDF vectorization and Logistic Regression to achieve accurate toxic comment detection.

## 🎯 Features

- **Russian Language Support**: Specialized tokenization and preprocessing for Russian text
- **Advanced Text Preprocessing**: 
  - Tokenization using NLTK
  - Stop word removal (Russian stopwords)
  - Stemming using Snowball Stemmer
  - Punctuation removal
- **Machine Learning Pipeline**: TF-IDF vectorization + Logistic Regression
- **Model Evaluation**: Precision, Recall, and Precision-Recall curve analysis
- **Threshold Optimization**: Customizable decision thresholds for improved precision

## 📊 Dataset

- **Size**: ~27,857 Russian comments
- **Format**: CSV file with two columns:
  - `comment`: Russian text comments
  - `toxic`: Binary labels (0 = non-toxic, 1 = toxic)
- **Split**: 500 samples reserved for testing, remainder for training

## 🛠️ Technologies Used

- **Python 3.11+**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **NLTK**: Natural language processing toolkit
- **matplotlib**: Data visualization
- **numpy**: Numerical computing

### Key Libraries:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import numpy as np
```

## 🚀 Getting Started

### Prerequisites

1. **Python 3.11+** installed on your system
2. **Required Python packages** (install via pip):

```bash
pip install -r requirements.txt
```

3. **NLTK Data** (automatically downloaded by the scripts)

### Installation & Usage

#### Option 1: Using the Jupyter Notebook
1. Clone or download this repository
2. Ensure `data.csv` is in the same directory as the notebook
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Jupyter notebook: `russian_toxic_comment_detection.ipynb`

#### Option 2: Using the API (Recommended)
1. **Train the model**:
   ```bash
   python model_exporter.py
   ```
   This will create `toxic_comment_model.pkl`

2. **Start the API server**:
   ```bash
   python api.py
   ```
   The API will be available at `http://localhost:8000`

3. **View API documentation**:
   Open `http://localhost:8000/docs` in your browser for interactive API docs

## 🌐 API Usage

The API provides several endpoints for toxicity detection:

### API Endpoints

- **Health Check**: `GET /health` - Check API status and model health
- **Single Prediction**: `POST /predict` - Analyze one text
- **Batch Prediction**: `POST /predict/batch` - Analyze multiple texts
- **Model Info**: `GET /model/info` - Get model details
- **Documentation**: `GET /docs` - Interactive API documentation

### Example API Usage

**Single Text Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Привет, как дела?"}'
```

**Response:**
```json
{
  "text": "Привет, как дела?",
  "is_toxic": false,
  "toxic_probability": 0.4717,
  "non_toxic_probability": 0.5283,
  "timestamp": "2025-01-31T11:55:00"
}
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Привет всем!", "идиот дурак", "хороший день"]}'
```

**Python Client Example:**
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "ваш русский текст здесь"}
)
result = response.json()
print(f"Toxic: {result['is_toxic']}, Confidence: {result['toxic_probability']:.3f}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["текст 1", "текст 2", "текст 3"]}
)
results = response.json()
print(f"Processed {results['total_count']} texts")
print(f"Toxic: {results['toxic_count']}, Non-toxic: {results['non_toxic_count']}")
```

## 📈 Model Performance

The system implements comprehensive evaluation metrics:

- **Accuracy**: 86.4% on test set
- **Precision**: 91.7% (high accuracy for positive predictions)
- **Recall**: 65.5% (completeness of toxic comment detection)
- **Vocabulary Size**: 35,254 unique Russian word stems
- **Training Set**: ~13,912 Russian comments
- **Test Set**: 500 Russian comments

### Model Architecture

- **Vectorization**: TF-IDF with custom Russian tokenizer
- **Preprocessing**: Tokenization, stemming, stop word removal
- **Classification**: Logistic Regression
- **Pipeline**: Integrated preprocessing + classification

## 🔧 Text Preprocessing Pipeline

The system includes a sophisticated Russian text preprocessing function:

```python
def tokenaiz_sentens(sentence: str, remove_stop_words: bool = True):
    # Tokenize using Russian language model
    tokens = word_tokenize(sentence, language="russian")
    
    # Remove punctuation
    tokens = [i for i in tokens if i not in string.punctuation]
    
    # Remove Russian stopwords (optional)
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    
    # Apply stemming
    tokens = [snowball.stem(i) for i in tokens]
    
    return tokens
```

## 📁 Project Structure

```
Data Science/
│
├── data.csv                              # Dataset with Russian comments and toxicity labels (27,857 comments)
├── russian_toxic_comment_detection.ipynb # Original Jupyter notebook implementation
├── toxic_comment_model.pkl               # Trained model (generated by model_exporter.py)
│
├── tokenizer.py                          # Shared Russian tokenizer module
├── model_exporter.py                     # Script to train and export the model
├── model_utils.py                        # Model loading and utility functions
├── api.py                                # FastAPI server for model serving
│
├── test_api.py                           # API testing script
├── test_model_loading.py                 # Model loading debug script
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation (this file)
```

## 🎯 Use Cases

- **Content Moderation**: Automatically filter toxic comments on social media platforms
- **Community Management**: Help maintain healthy online discussions
- **Research**: Study patterns in toxic behavior in Russian-language online communities
- **Educational**: Learn about NLP techniques for non-English languages
- **API Integration**: Embed toxicity detection into existing applications via REST API

## 🚀 Production Deployment

For production use, consider these recommendations:

### Performance Optimization
```bash
# Use gunicorn for production serving
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000

# Or use Docker (create Dockerfile)
# FROM python:3.11-slim
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY . .
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Security & Monitoring
- Add rate limiting for API endpoints
- Implement authentication/authorization as needed
- Monitor API usage and model performance
- Set up logging and error tracking
- Consider input sanitization and validation

### Scaling Considerations
- Use load balancers for high traffic
- Consider model caching for frequently analyzed texts
- Implement async processing for batch operations
- Monitor memory usage (model is ~1.2MB)

## ⚠️ Important Notes

- The model is specifically trained for **Russian language** text
- Requires NLTK Russian language data for optimal performance
- Dataset contains real user comments with potentially offensive content
- Model performance can be adjusted via threshold optimization
- **Production Use**: Always implement human oversight for content moderation decisions

## 🤝 Contributing

This project is open for improvements. Potential areas for enhancement:

### Model Improvements
- Deep learning models (BERT, RoBERTa for Russian)
- Multi-class classification (different types of toxicity)
- Threshold optimization for better precision/recall balance
- Cross-validation and hyperparameter tuning

### Technical Enhancements
- Docker containerization
- Kubernetes deployment manifests
- CI/CD pipeline setup
- Performance benchmarking
- Automated testing suite

### Features
- Web interface development
- Real-time inference optimization
- Multi-language support
- Confidence score explanations

## 📄 License

This project is available for educational and research purposes. Please ensure appropriate use when handling sensitive content.

---

**Note**: This system is designed for research and educational purposes. When deploying in production environments, please implement additional safeguards, human oversight, and proper content moderation policies.