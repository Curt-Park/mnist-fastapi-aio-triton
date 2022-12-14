"""Locus file for load tests.

- Author: Jinwoo Park
- Email: www.jwpark.co.kr@gmail.com

Reference:
    http://docs.locust.io/en/stable/writing-a-locustfile.html
    http://docs.locust.io/en/stable/increase-performance.html
    http://docs.locust.io/en/stable/running-distributed.html
"""
import base64
import cv2

from typing import Any
from locust import FastHttpUser, constant, task
from src.api.utils import encode_img_base64


class APIUser(FastHttpUser):
    """Send requests."""

    wait_time = constant(1)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize."""
        super().__init__(*args, **kwargs)
        img_path = "mnist_sample.jpg"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape[:2]
        img_enc = encode_img_base64(img)
        self.req = {"image": img_enc, "height": height, "width": width}

    @task
    def predict_digits(self) -> None:
        """Request mnist prediction."""
        self.client.get("/predict-mnist", json=self.req)
