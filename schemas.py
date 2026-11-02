from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class InputProductData(BaseModel):
    name: str = Field(..., min_length=2, max_length=250)
    manufacturer: str = Field(..., min_length=1, max_length=150)
    specs: Dict[str, str] = Field(default_factory=dict)
    description: str = Field(..., min_length=5)

    @field_validator('name', 'manufacturer', 'description', mode='before')
    @classmethod
    def strip_and_sanitize(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip()
            v = v.replace('\x00', '')
            return v
        return v

    @field_validator('specs', mode='before')
    @classmethod
    def sanitize_specs(cls, v: object) -> object:
        if not isinstance(v, dict):
            raise ValueError("specs must be an object with string keys and values")
        cleaned: Dict[str, str] = {}
        for key, val in v.items():
            if not isinstance(key, str) or not isinstance(val, str):
                raise ValueError("specs must contain only string keys and values")
            cleaned_key = key.strip().replace('\x00', '')
            cleaned_val = val.strip().replace('\x00', '')
            if cleaned_key and cleaned_val:
                cleaned[cleaned_key] = cleaned_val
        return cleaned

class OutputProductCard(BaseModel):
    missing_fields_warning: Optional[str] = None
    seo_title: str = Field(..., max_length=70)
    seo_description: str = Field(..., min_length=800, max_length=1200)
    structured_specs: Dict[str, str]
    search_queries: List[str] = Field(..., min_length=3, max_length=5)
    commercial_block: str