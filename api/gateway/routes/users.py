"""Users routes."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class UserProfile(BaseModel):
    id: str
    name: str

@router.get("/me")
def get_me():
    return {"id": "1", "name": "User"}
