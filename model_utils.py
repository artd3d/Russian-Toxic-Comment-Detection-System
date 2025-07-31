"""
Utility functions for Russian Toxic Comment Detection Model
Contains preprocessing functions and model loading utilities
"""

import pickle
import os
from typing import List, Tuple

# Import the shared tokenizer
from tokenizer import RussianTokenizer, tokenaiz_sentens

def load_model(model_path: str = 'toxic_comment_model.pkl'):
    """
    Load the trained model from disk
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        sklearn.pipeline.Pipeline: Loaded model pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_toxicity(model, text: str) -> Tuple[bool, float, float]:
    """
    Predict toxicity for a given text
    
    Args:
        model: Trained model pipeline
        text (str): Input text to analyze
        
    Returns:
        Tuple[bool, float, float]: (is_toxic, non_toxic_probability, toxic_probability)
    """
    try:
        # Make prediction
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        is_toxic = bool(prediction == 1)
        non_toxic_prob = float(probabilities[0])
        toxic_prob = float(probabilities[1])
        
        return is_toxic, non_toxic_prob, toxic_prob
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

def batch_predict_toxicity(model, texts: List[str]) -> List[Tuple[bool, float, float]]:
    """
    Predict toxicity for multiple texts
    
    Args:
        model: Trained model pipeline
        texts (List[str]): List of texts to analyze
        
    Returns:
        List[Tuple[bool, float, float]]: List of (is_toxic, non_toxic_probability, toxic_probability)
    """
    try:
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            is_toxic = bool(pred == 1)
            non_toxic_prob = float(prob[0])
            toxic_prob = float(prob[1])
            results.append((is_toxic, non_toxic_prob, toxic_prob))
        
        return results
    
    except Exception as e:
        raise Exception(f"Error during batch prediction: {str(e)}")

def validate_text_input(text: str) -> str:
    """
    Validate and clean text input
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    text = text.strip()
    
    if not text:
        raise ValueError("Input text cannot be empty")
    
    if len(text) > 10000:  # Reasonable limit
        raise ValueError("Input text too long (max 10000 characters)")
    
    return text

def get_model_info(model) -> dict:
    """
    Get information about the loaded model
    
    Args:
        model: Trained model pipeline
        
    Returns:
        dict: Model information
    """
    try:
        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['model']
        
        info = {
            "model_type": "Pipeline",
            "vectorizer": type(vectorizer).__name__,
            "classifier": type(classifier).__name__,
            "vocabulary_size": len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else "Not available",
            "feature_names_count": len(vectorizer.get_feature_names_out()) if hasattr(vectorizer, 'get_feature_names_out') else "Not available"
        }
        
        return info
    
    except Exception as e:
        return {"error": f"Could not retrieve model info: {str(e)}"}