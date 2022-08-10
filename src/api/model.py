"""Data formats for predictions."""
from pydantic import BaseModel, Field


class ImageReq(BaseModel):
    """Image Request model."""

    image: str = Field(
        ..., title="utf-8 string from a base64 encoded image", example="AQIDBAUG"
    )
    height: int = Field(..., title="Image's height", example=1)
    width: int = Field(..., title="Image's width", example=2)
