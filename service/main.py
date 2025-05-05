"""API module with API Server."""

from fastapi import Depends, FastAPI
from starlette.status import HTTP_200_OK
from fastapi.middleware.cors import CORSMiddleware
from service.router import app_router



app = FastAPI(
    title="Sam2 Encoder Fast API",
    description="",
    docs_url="/docs",
)

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app.include_router(app_router)

@app.get("/", status_code=HTTP_200_OK)
def root():
    """Root API path.

    Returns
    -------
    dict
        Returns service environment and version
    """
    print("Get request at /")
    return {
        "message": "sam2-encoder-api",
    }


