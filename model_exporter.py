"""
Model Exporter for Russian Toxic Comment Detection
This script trains the model and saves it for use in the API
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Import the shared tokenizer
from tokenizer import RussianTokenizer, tokenaiz_sentens

def train_and_save_model():
    """
    Train the toxic comment detection model and save it to disk
    """
    print("Loading dataset...")
    
    # Load the dataset
    if not os.path.exists('data.csv'):
        raise FileNotFoundError("data.csv not found. Please ensure the dataset is in the current directory.")
    
    df = pd.read_csv('data.csv', sep=',')
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Convert toxic column to int
    df['toxic'] = df['toxic'].apply(int)
    print(f"Toxic comments distribution:\n{df['toxic'].value_counts()}")
    
    # Split the data
    train_df, test_df = train_test_split(df, test_size=500, random_state=42)
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Create the model pipeline
    print("Creating model pipeline...")
    tokenizer = RussianTokenizer(remove_stop_words=True)
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=tokenizer)),
        ('model', LogisticRegression(random_state=0))
    ])
    
    # Train the model
    print("Training the model...")
    model_pipeline.fit(train_df['comment'], train_df['toxic'])
    print("Model training completed!")
    
    # Test the model
    print("Testing model predictions...")
    test_predictions = model_pipeline.predict(test_df['comment'])
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    
    accuracy = accuracy_score(test_df['toxic'], test_predictions)
    precision = precision_score(test_df['toxic'], test_predictions)
    recall = recall_score(test_df['toxic'], test_predictions)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Save the model
    model_filename = 'toxic_comment_model.pkl'
    print(f"Saving model to {model_filename}...")
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model_pipeline, f)
    
    print(f"Model saved successfully to {model_filename}")
    
    # Test saved model
    print("Testing saved model...")
    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Test predictions
    test_texts = [
        "привет у меня все нормально",
        "тебе что пофиг на всех",
        "хорошая погода сегодня"
    ]
    
    print("Sample predictions:")
    for text in test_texts:
        prediction = loaded_model.predict([text])[0]
        probability = loaded_model.predict_proba([text])[0]
        print(f"Text: '{text}'")
        print(f"Prediction: {'Toxic' if prediction == 1 else 'Non-toxic'}")
        print(f"Probabilities: Non-toxic={probability[0]:.4f}, Toxic={probability[1]:.4f}")
        print("-" * 50)
    
    return model_pipeline

if __name__ == "__main__":
    try:
        model = train_and_save_model()
        print("Model export completed successfully!")
    except Exception as e:
        print(f"Error during model export: {str(e)}")
        raise