# main.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException as FastAPIHTTPException # Alias to avoid confusion
import os

# Import routers from separate API files
from api.slp_api import router as slp_router
from api.audio_api import router as audio_router

# Import PostgreSQL database connection functions
from services.database import connect_db, close_db
from models import MessageResponse # Import MessageResponse for consistent error formatting


# --- FastAPI App Initialization ---
app = FastAPI(
    title="SpeechAid Backend API",
    description="API for Speech-Language Pathologist (SLP) and Patient Management",
    version="1.0.0",
)

# CORS middleware for allowing frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    # UPDATED: Explicitly list allowed origins when allow_credentials=True
    allow_origins=[
        "http://localhost:5174",          # For local frontend dev without Nginx
        "http://localhost:5173",
        "http://192.168.0.49:5021",       # For local frontend dev via internal IP
        "http://192.168.0.49",           # For Nginx proxying to frontend on 443 (if accessing Nginx via internal IP)
        "https://38.188.108.234",  
        "https://38.188.108.234:5021",     # For Nginx proxying to frontend on 443 (if accessing Nginx via public IP)
        # Add your actual domain if you switch back to it for production
        # "https://speechaid.convogene.ai",
        "http://localhost:8001",          # If you ever run backend directly from localhost
        "http://192.168.0.49:8001",        # If you ever run backend directly from internal IP
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handler for FastAPI HTTPExceptions
@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    """
    Handles HTTPExceptions raised in the application, formatting them into a MessageResponse.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=MessageResponse(
            message="HTTP Error",
            detail=exc.detail,
            success=False
        ).dict() # Convert Pydantic model to dictionary
    )

# Include API routers
app.include_router(slp_router, tags=["SLP & Patients"])
app.include_router(audio_router, tags=["Audio Upload"])

@app.get("/")
async def read_root():
    """Root endpoint for basic API check."""
    return {"message": "Welcome to SpeechAid Backend!"}

@app.on_event("startup")
async def startup_event():
    print("Application starting up...")
    await connect_db()

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutting down...")
    await close_db()

