from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, HttpUrl


class BoundarySource(BaseModel):
    code_authority: str
    geometry_authority: str
    code_url: HttpUrl
    geometry_url: HttpUrl


class UpcomingBoundaryChange(BaseModel):
    id: str
    district: str
    title: str
    change_type: Literal["rename", "split", "merge", "boundary_adjustment", "unknown"]
    announced_at: date
    effective_from: date | None = None
    status: Literal["detected", "confirmed", "scheduled"]
    source_url: HttpUrl


class AdminBoundaryStatus(BaseModel):
    applied_version: str
    applied_year_month: str
    effective_from: date
    verified_at: datetime
    district_count: int
    neighborhood_count: int
    boundary_file: str
    status: Literal["applied"]
    source: BoundarySource
    upcoming_changes: list[UpcomingBoundaryChange]
