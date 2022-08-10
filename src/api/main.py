"""API server that accepts requests and returns values."""
import logging
import logging.config
import os

from fastapi import FastAPI

from .predictor import DigitPredictor
from .model import ImageReq, PredictionRes

if not os.path.exists("logs"):
    os.mkdir("logs")

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

app = FastAPI()
digit_predictor = DigitPredictor()


@app.get("/")
def healthcheck() -> bool:
    """Check the server's status."""
    return True


@app.get("/predict-mnist", response_model=PredictionRes)
async def predict_digit(data: ImageReq) -> PredictionRes:
    """Predict the digit from the input image."""
    logger.info("Received an image")
    prediction = await digit_predictor.predict(data.image, data.height, data.width)
    logger.info("Predicted %d", prediction)
    return PredictionRes(prediction=prediction)
