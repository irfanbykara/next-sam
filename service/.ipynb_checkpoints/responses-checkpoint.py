"""Response Objects"""
from pydantic import BaseModel

class BaseResponse(BaseModel):
    response: str
    status_code: int = 200

class Sam2EncoderResponse(BaseResponse):
    response: str = "Success Sam2 Encoder Endpoint"
    high_res_feats_0: str
    high_res_feats_1: str
    image_embed: str
