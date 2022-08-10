"""API server that accepts requests and returns values."""
import logging
import logging.config
import os

from fastapi import FastAPI

from .predictor import DigitPredictor

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


@app.get("/predict/mnist")
async def predict_digit(image: str) -> int:
    """Predict the digit from the input image."""
    logger.info("Received an image")
    return await digit_predictor.predict(image)
