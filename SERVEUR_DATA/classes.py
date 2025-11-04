from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict, Any


class DatasetIn(BaseModel):
    # name obligatoire et UNIQUE (vérifié dans l'endpoint)
    name: str = Field(..., description="Nom lisible (obligatoire et unique)")
    dates: Optional[List[Union[str, None]]] = None
    timestamps: Optional[List[Union[str, float, int, None]]] = None
    values: List[Union[float, int, None]] = Field(..., description="Liste des valeurs; null accepté (sera filtré).")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("name est obligatoire et ne doit pas être vide.")
        return v

    @field_validator("dates", "timestamps", mode="before")
    @classmethod
    def empty_to_none(cls, v):
        return v if v not in ([], (), "", None) else None

    @field_validator("values")
    @classmethod
    def values_not_empty(cls, v: List[Union[float, int, None]]):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("values ne doit pas être vide.")
        return v

class DatasetOut(BaseModel):
    id: str
    name: Optional[str] = None
    n_points: int
    time_kind: Optional[str] = None   # "dates" | "timestamps" | None
    preview: Dict[str, Any]
    meta: Dict[str, Any]