"""Models module."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class Model(BaseModel):
    """Base model."""
    id: str = ""

class User(BaseModel):
    """User model."""
    id: str = ""
    name: str = ""
    email: str = ""

class Item(BaseModel):
    """Item model."""
    id: str = ""
    name: str = ""
