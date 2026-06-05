import numpy as np
import logging

try:
    import tritonclient.grpc as grpcclient
except ImportError:
    grpcclient = None

logger = logging.getLogger(__name__)

class TritonRerankerClient:
    def __init__(self, url: str = "localhost:8001", model_name: str = "reranker"):
        self.url = url
        self.model_name = model_name
        self.client = None
        if grpcclient:
            self.client = grpcclient.InferenceServerClient(url=url)
        else:
            logger.warning("tritonclient[grpc] not installed. TritonRerankerClient will not work.")

    def is_available(self) -> bool:
        if not self.client:
            return False
        try:
            return self.client.is_server_live()
        except Exception:
            return False

    def rerank(self, features: np.ndarray) -> np.ndarray:
        """
        Calls Triton Inference Server for reranking via gRPC.
        Args:
            features: (B, input_dim) numpy array
        Returns:
            scores: (B, 1) numpy array
        """
        if not self.client:
            raise RuntimeError("Triton gRPC client not initialized")

        inputs = [
            grpcclient.InferInput("input", features.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(features.astype(np.float32))

        outputs = [
            grpcclient.InferRequestedOutput("output")
        ]

        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )

        return response.as_numpy("output")
