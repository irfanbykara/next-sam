"""API Service router."""

from fastapi import APIRouter
from starlette.status import HTTP_200_OK
from service.endpoint import sam2_encoder


app_router = APIRouter(
    prefix="/main",
    tags=["v1"]
)

app_router.add_api_route(
    methods=["POST"],
    path="/sam2_encoder",
    endpoint=sam2_encoder,
    response_model=dict,
    status_code=HTTP_200_OK,
    summary="Encode images with SAM2",
    description="Processes images for advanced segmentation tasks."
)

