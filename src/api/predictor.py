"""Async Triton predictor."""
import os
import tritonclient.grpc.aio as grpcclient
import numpy as np

from .utils import decode_img


class ImageClassifier:
    """Image classifier that returns an index as a classification result."""

    def __init__(self) -> None:
        """Initialize."""
        url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
        self.triton_client = grpcclient.InferenceServerClient(url=url)
        self.outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]

    async def predict(
        self, img: str, height: int, width: int, model_name: str = "mnist_cnn"
    ) -> int:
        """Predict the digit from the image encoded w/ base64."""
        img = decode_img(img)
        img = img.reshape(1, 1, height, width)

        # predict w/ triton
        inputs = [grpcclient.InferInput("INPUT__0", img.shape, "FP32")]
        inputs[0].set_data_from_numpy(img.astype(np.float32))
        results = await self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=self.outputs
        )

        # get the result from triton
        output = results.as_numpy("OUTPUT__0")
        top_1 = output.argmax()
        return top_1
