"""
Test script to debug model loading issues
"""

import pickle
import sys

print("Testing model loading...")

try:
    print("1. Attempting to load model...")
    with open('toxic_comment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
    
    print("2. Testing model prediction...")
    test_text = "Привет, как дела?"
    prediction = model.predict([test_text])
    probability = model.predict_proba([test_text])
    
    print(f"✅ Prediction successful!")
    print(f"   Text: '{test_text}'")
    print(f"   Prediction: {prediction[0]}")
    print(f"   Probabilities: {probability[0]}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    print("   Full traceback:")
    traceback.print_exc()

print("\nTesting imports...")
try:
    from model_utils import load_model
    print("✅ model_utils import successful")
    
    model2 = load_model()
    print("✅ Model loaded via model_utils")
    
    prediction2 = model2.predict(["тест"])
    print("✅ Prediction via model_utils successful")
    
except Exception as e:
    print(f"❌ model_utils error: {str(e)}")
    import traceback
    traceback.print_exc()