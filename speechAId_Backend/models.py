# models.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any

class SLPSignupRequest(BaseModel):
    slp_id: str
    email: EmailStr
    name: str

class PatientRegistrationRequest(BaseModel):
    slp_id: str
    name: str
    age: int
    gender: str
    ailment: str
    severity: str
    clinic_name: Optional[str] = None

class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None
    success: bool

# New model for LLM phrase generation response
class GeneratePhrasesResponse(BaseModel):
    phrases: List[str]
    message: str
    success: bool
    detail: Optional[str] = None

