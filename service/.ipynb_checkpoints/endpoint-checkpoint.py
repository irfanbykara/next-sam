from service.requests import Sam2EncoderRequest
from fastapi.responses import JSONResponse
from service.inference import Sam2Endpoint

sam2_endpoint = Sam2Endpoint()

def sam2_encoder(request: Sam2EncoderRequest) -> dict:
    """
     Create the image embeddings for manual segmentation using Sam2.1 Large Model
    Parameters:
        request (Sam2EncoderRequest): The request containing the input image.

    Returns:
        dict: Response containing a dictionary of high_res_feats_0,high_res_feats_1 and image_embed
    Raises:
        Exception: If an error occurs during the Sam2Encoder image answering api call process.
    """

    try:
        response = sam2_endpoint.sam2_encoder(request)
        return response.model_dump()

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error : {str(e)}"}
        )