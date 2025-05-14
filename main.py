from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Item(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "I love using FastAPI!"
            }
        }


class PredictionResult(BaseModel):
    label: str
    score: float


app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using Hugging Face pipeline"
)

# Initialize model
try:
    classifier = pipeline("sentiment-analysis")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint returning welcome message"""
    return {"message": "Hello World"}


@app.post("/predict/", response_model=PredictionResult, tags=["Prediction"])
async def predict(item: Item):
    """
    Predict sentiment of input text

    - **text**: input text to analyze (min 2 characters)
    """
    try:
        if len(item.text.strip()) < 2:
            raise HTTPException(
                status_code=422,
                detail="Text must be at least 2 characters long"
            )

        result = classifier(item.text)[0]
        logger.info(f"Prediction successful for text: {item.text[:50]}...")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error processing your request"
        )
