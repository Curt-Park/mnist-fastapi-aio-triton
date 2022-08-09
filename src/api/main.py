"""API server that accepts requests and returns values."""
import logging
import logging.config
import os

from fastapi import FastAPI

from predictor import digit_predictor

if not os.path.exists("logs"):
    os.mkdir("logs")

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()
app = FastAPI()


@app.get("/")
def root() -> str:
    """Show the proper usage."""
    return "Try api-address:api-port/predict/image_name"


@app.get("/predict")
def predict_without_image() -> str:
    """Show the proper usage."""
    return "Try api-address:api-port/predict/image_name"


@app.get("/predict/mnist")
async def predict_digit(image: str) -> int:
    """Predict the digit from the input image."""
    logger.info("Received an image")
    return await digit_predictor.predict(image)
