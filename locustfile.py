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
from locust import FastHttpUser, task
from src.api.utils import encode_img_base64


class APIUser(FastHttpUser):
    """Send requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize."""
        super().__init__(*args, **kwargs)
        img_path = "mnist_sample.jpg"
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        img_enc = encode_img_base64(img)
        self.req = {"image": img}

    @task
    def predict_digits(self) -> None:
        """Request mnist prediction."""
        # send
        self.client.get("/predict/mnist", self.req)
