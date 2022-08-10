"""Async Triton predictor."""
import os
import tritonclient.grpc.aio as grpcclient

from .utils import decode_img


class DigitPredictor:
    """Predictor for digits w/ images."""

    def __init__(self) -> None:
        """Initialize."""
        url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
        self.triton_client = grpcclient.InferenceServerClient(url=url)
        self.outputs = [grpcclient.InferRequestedOutput("OUTPUT")]
        self.model_name = "mnist_cnn"

    async def predict(self, img: str) -> int:
        """Predict the digit from the image encoded w/ base64."""
        img = decode_img(img)

        # predict w/ triton
        inputs = [grpcclient.InferInput("INPUT", image.shape, "FLOAT32")]
        inputs[0].set_data_from_numpy(image)
        results = await triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=self.outputs
        )

        # get the result from triton
        output = response.as_numpy("OUTPUT")
        print(output)
        return 1
