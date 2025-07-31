"""
FastAPI Server for Russian Toxic Comment Detection
Serves the trained machine learning model via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging
import sys
from datetime import datetime

# Import our model utilities
from model_utils import (
    load_model, 
    predict_toxicity, 
    batch_predict_toxicity, 
    validate_text_input,
    get_model_info
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Russian Toxic Comment Detection API",
    description="Machine Learning API for detecting toxic comments in Russian text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str = Field(..., description="Russian text to analyze for toxicity", example="Привет, как дела?")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of Russian texts to analyze", example=["Привет!", "Плохой день"])

class ToxicityPrediction(BaseModel):
    text: str
    is_toxic: bool
    toxic_probability: float
    non_toxic_probability: float
    timestamp: str

class BatchToxicityPrediction(BaseModel):
    predictions: List[ToxicityPrediction]
    total_count: int
    toxic_count: int
    non_toxic_count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    model_info: Optional[dict] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global model
    try:
        logger.info("Loading toxic comment detection model...")
        model = load_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error("API will not function properly without the model!")
        # Don't exit - let the health endpoint show the error

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health"""
    model_info = None
    if model is not None:
        try:
            model_info = get_model_info(model)
        except Exception as e:
            logger.warning(f"Could not get model info: {str(e)}")
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        model_info=model_info
    )

# Single text prediction endpoint
@app.post("/predict", response_model=ToxicityPrediction)
async def predict_single_text(input_data: TextInput):
    """Predict toxicity for a single Russian text"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Validate input
        cleaned_text = validate_text_input(input_data.text)
        
        # Make prediction
        is_toxic, non_toxic_prob, toxic_prob = predict_toxicity(model, cleaned_text)
        
        return ToxicityPrediction(
            text=cleaned_text,
            is_toxic=is_toxic,
            toxic_probability=toxic_prob,
            non_toxic_probability=non_toxic_prob,
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchToxicityPrediction)
async def predict_batch_texts(input_data: BatchTextInput):
    """Predict toxicity for multiple Russian texts"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if len(input_data.texts) > 100:  # Reasonable batch limit
        raise HTTPException(
            status_code=400, 
            detail="Batch size too large. Maximum 100 texts allowed."
        )
    
    try:
        # Validate all inputs
        cleaned_texts = []
        for text in input_data.texts:
            cleaned_texts.append(validate_text_input(text))
        
        # Make batch prediction
        results = batch_predict_toxicity(model, cleaned_texts)
        
        # Format response
        predictions = []
        toxic_count = 0
        timestamp = datetime.now().isoformat()
        
        for text, (is_toxic, non_toxic_prob, toxic_prob) in zip(cleaned_texts, results):
            predictions.append(ToxicityPrediction(
                text=text,
                is_toxic=is_toxic,
                toxic_probability=toxic_prob,
                non_toxic_probability=non_toxic_prob,
                timestamp=timestamp
            ))
            if is_toxic:
                toxic_count += 1
        
        return BatchToxicityPrediction(
            predictions=predictions,
            total_count=len(predictions),
            toxic_count=toxic_count,
            non_toxic_count=len(predictions) - toxic_count
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Model information endpoint
@app.get("/model/info")
async def get_model_information():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        info = get_model_info(model)
        return {
            "model_info": info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Russian Toxic Comment Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "model_info": "/model/info"
        }
    }

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("Starting Russian Toxic Comment Detection API...")
    print("Model will be loaded on startup...")
    print("API Documentation available at: http://localhost:8000/docs")
    
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )