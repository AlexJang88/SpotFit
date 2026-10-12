from typing import Literal

from pydantic import BaseModel


class SpotFitResponse(BaseModel):
    status: Literal["ok"]
