# api/llm_api.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import httpx # For making asynchronous HTTP requests

router = APIRouter()

# Pydantic model for request body
class GeneratePhrasesRequest(BaseModel):
    topic: str

# Pydantic model for response body
class GeneratePhrasesResponse(BaseModel):
    phrases: List[str]
    message: str
    success: bool
    detail: Optional[str] = None

# Gemini API configuration
# In a real application, the API key would be loaded securely from environment variables
# or a secrets manager, and not hardcoded. For the purpose of this demo, it's left empty.
GEMINI_API_KEY = "" # Canvas environment will provide this at runtime if empty
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="

@router.post("/generate_phrases", response_model=GeneratePhrasesResponse)
async def generate_phrases_from_llm(request_body: GeneratePhrasesRequest):
    """
    Generates a list of speech therapy phrases based on a given topic using the Gemini API.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API Key is not configured.",
            success=False
        )

    prompt = f"""You are an AI assistant specialized in generating speech therapy practice phrases. A Speech-Language Pathologist needs phrases related to the topic: '{request_body.topic}'. Generate 10 distinct, simple, and commonly used phrases or short sentences that a patient could practice for speech articulation or fluency. Each phrase should be on a new line. Do not include any introductory or concluding remarks, just the phrases.
    """

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": { "type": "STRING" }
            }
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GEMINI_API_URL}{GEMINI_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0 # Set a timeout for the API call
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            gemini_result = response.json()

            if gemini_result.get("candidates") and gemini_result["candidates"][0].get("content") and gemini_result["candidates"][0]["content"].get("parts"):
                # The response from structured generation is often a stringified JSON array
                raw_json_string = gemini_result["candidates"][0]["content"]["parts"][0]["text"]
                generated_phrases_list = json.loads(raw_json_string) # Parse the stringified JSON array

                if not isinstance(generated_phrases_list, list):
                    raise ValueError("LLM response was not a list of phrases.")

                return GeneratePhrasesResponse(
                    phrases=generated_phrases_list,
                    message="Phrases generated successfully.",
                    success=True
                )
            else:
                return GeneratePhrasesResponse(
                    phrases=[],
                    message="No candidates or content found in Gemini API response.",
                    success=False,
                    detail=f"Full response: {gemini_result}"
                )

    except httpx.HTTPStatusError as e:
        print(f"HTTP error generating phrases: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Gemini API HTTP Error: {e.response.text}",
            success=False
        )
    except httpx.RequestError as e:
        print(f"Network error generating phrases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Network error when calling Gemini API: {e}",
            success=False
        )
    except json.JSONDecodeError as e:
        print(f"JSON decode error from Gemini API: {e}. Raw text: {raw_json_string}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid JSON response from Gemini API: {e}",
            success=False
        )
    except ValueError as e:
        print(f"Value error from Gemini API response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected response structure from Gemini API: {e}",
            success=False
        )
    except Exception as e:
        print(f"An unexpected error occurred during phrase generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}",
            success=False
        )

