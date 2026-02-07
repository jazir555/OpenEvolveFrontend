"""Models Schemas module."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel

class User(BaseModel):
    """User model."""
    id: str = ""
    name: str = ""
    email: str = ""

class Item(BaseModel):
    """Item model."""
    id: str = ""
    name: str = ""
    description: str = ""

class schemas:
    """Schemas namespace."""
    User = User
    Item = Item
