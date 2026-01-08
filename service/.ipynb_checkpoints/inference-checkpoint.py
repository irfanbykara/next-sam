from service.requests import Sam2EncoderRequest
from service.responses import Sam2EncoderResponse
from service.sam2 import SAM2Onnx
from io import BytesIO
import base64
from PIL import Image

class Sam2Endpoint:
    def __init__(self):
        self.sam2_encoder_session = SAM2Onnx()

    def _base64_to_pil(self, image_base64):
        try:
            return Image.open(BytesIO(base64.b64decode(image_base64)))
        except Exception as e:
            print(e)

    def sam2_encoder(
            self,
            request: Sam2EncoderRequest
    ) -> Sam2EncoderResponse:
        try:

            image_base64 = request.image_base64
            image = self._base64_to_pil(image_base64)
            inputs = self.sam2_encoder_session.prepare_inputs(image)
            results = self.sam2_encoder_session.session_encoder.run(None, inputs)

            return Sam2EncoderResponse(
                response="Success Modelia SAM2 Encoder Response",
                high_res_feats_0=self.sam2_encoder_session.numpy_to_base64(results[0]),
                high_res_feats_1=self.sam2_encoder_session.numpy_to_base64(results[1]),
                image_embed=self.sam2_encoder_session.numpy_to_base64(results[2])
            )

        except Exception as e:
            print(e)
