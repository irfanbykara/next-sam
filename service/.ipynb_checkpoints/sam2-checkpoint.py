import onnxruntime as ort
from huggingface_hub import snapshot_download, repo_exists
import base64
import numpy as np
import os


CKPT_FOLDER = "CKPT_FOLDER"

class SAM2Onnx:

    def __init__(self, repo_root=None,repo_name="sam2.1", encoder_name="sam2.1_hiera_large_encoder.with_runtime_opt.ort"):

        self.repo_root = repo_root
        self.repo_name = repo_name
        self.encoder_name = encoder_name
        self.repo_path = os.path.join(CKPT_FOLDER, self.repo_name)
        self.encoder_path = os.path.join(self.repo_path, self.encoder_name)

        self.session_encoder = None
        self.image_encoded = None
        self._ensure_model_exists()
        self.create_sessions()

    def create_sessions(self):
        try:
            self.session_encoder = ort.InferenceSession(self.encoder_path, providers=["CUDAExecutionProvider"])
            print("Active Execution Providers:", self.session_encoder.get_providers())

            print("CUDA is available for encoding")
        except:
            self.session_encoder = ort.InferenceSession(self.encoder_path, providers=["CPUExecutionProvider"])
            print("Using CPU for encoding")

    def _ensure_model_exists(self):
        if not os.path.exists(self.encoder_path):
            # Download everything from the repo to the desired path
            local_dir = snapshot_download(repo_id=f"{self.repo_root}/{self.repo_name}", local_dir=self.repo_path)

    @staticmethod
    def prepare_inputs(image):

        image = image.convert("RGB")
        # Convert image to numpy array and normalize to [0, 1] range
        input_array = np.array(image).astype(np.float32) / 255.0

        input_tensor = np.transpose(input_array, (2, 0, 1))  # Convert to (3, 1024, 1024)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension: (1, 3, 1024, 1024)

        # Run ONNX encoder
        inputs = {"image": input_tensor}
        return inputs

    @staticmethod
    def numpy_to_base64(arr):
        return base64.b64encode(arr.tobytes()).decode("utf-8")  # Encode to base64 string

