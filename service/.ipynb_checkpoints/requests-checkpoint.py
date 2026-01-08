"""Request objects"""
from pydantic import BaseModel, Field


class Sam2EncoderRequest(BaseModel):
    """Request model for Sam2Encoder."""
    image_base64: str = Field(..., description="Base64 encoded image to be encoded.")


