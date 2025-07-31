"""
Test script for Russian Toxic Comment Detection API
Tests various endpoints and functionality
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the API endpoints"""
    print("Testing Russian Toxic Comment Detection API")
    print("=" * 50)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            if health_data.get('model_info'):
                print(f"   Model type: {health_data['model_info'].get('model_type', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    
    except requests.ConnectionError:
        print("❌ Cannot connect to API server. Is it running?")
        return False
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            root_data = response.json()
            print(f"✅ Root endpoint: {root_data['message']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {str(e)}")
    
    # Test single prediction endpoint
    print("\n3. Testing single prediction endpoint...")
    test_texts = [
        "Привет, как дела?",  # Non-toxic
        "тебе что пофиг на всех",  # Toxic (from model training)
        "хорошая погода сегодня",  # Non-toxic
        "дурак идиот",  # Likely toxic
        "спасибо за помощь"  # Non-toxic
    ]
    
    for text in test_texts:
        try:
            payload = {"text": text}
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                toxicity = "🔴 TOXIC" if result['is_toxic'] else "🟢 NON-TOXIC"
                confidence = result['toxic_probability'] if result['is_toxic'] else result['non_toxic_probability']
                print(f"   Text: '{text}'")
                print(f"   Result: {toxicity} (confidence: {confidence:.3f})")
                print()
            else:
                print(f"❌ Prediction failed for '{text}': {response.status_code}")
                print(f"   Error: {response.text}")
        
        except Exception as e:
            print(f"❌ Prediction error for '{text}': {str(e)}")
    
    # Test batch prediction endpoint
    print("\n4. Testing batch prediction endpoint...")
    try:
        batch_texts = [
            "Привет всем!",
            "идиот дурак",
            "хороший день для прогулки"
        ]
        payload = {"texts": batch_texts}
        response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful:")
            print(f"   Total texts: {result['total_count']}")
            print(f"   Toxic: {result['toxic_count']}")
            print(f"   Non-toxic: {result['non_toxic_count']}")
            
            for prediction in result['predictions']:
                toxicity = "🔴 TOXIC" if prediction['is_toxic'] else "🟢 NON-TOXIC"
                confidence = prediction['toxic_probability'] if prediction['is_toxic'] else prediction['non_toxic_probability']
                print(f"   '{prediction['text']}' -> {toxicity} ({confidence:.3f})")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
    
    # Test model info endpoint
    print("\n5. Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        if response.status_code == 200:
            info = response.json()
            print("✅ Model info retrieved:")
            if 'model_info' in info:
                for key, value in info['model_info'].items():
                    print(f"   {key}: {value}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Model info error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")
    return True

if __name__ == "__main__":
    success = test_api()
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("❌ Some tests failed. Check the server logs.")