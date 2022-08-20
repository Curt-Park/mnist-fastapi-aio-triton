"""API server that accepts requests and returns values."""
import logging
import logging.config
import os

from fastapi import FastAPI

from .predictor import ImageClassifier
from .model import ImageClassficationRequest, PredictionResponse

if not os.path.exists("logs"):
    os.mkdir("logs")

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

app = FastAPI()
classifier = ImageClassifier()


@app.get("/")
def healthcheck() -> bool:
    """Check the server's status."""
    return True


@app.get("/predict-mnist", response_model=PredictionResponse)
async def predict_digit(data: ImageClassficationRequest) -> PredictionResponse:
    """Predict the digit from the input image."""
    logger.info("Received an image")
    prediction = await classifier.predict(data.image, data.height, data.width)
    logger.info("Predicted %d", prediction)
    return PredictionResponse(prediction=prediction)
