# api/audio_api.py
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
import datetime
import uuid
import os
import json # Kept for potential future use or if other local data handling is added
import asyncpg
from pydantic import BaseModel
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

from models import MessageResponse
from services.database import get_db_connection

# Load environment variables from .env file
load_dotenv()

# S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET = os.getenv('S3_BUCKET')

s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_REGION and S3_BUCKET:
    try:
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print(f"S3 client initialized successfully within audio_api for bucket: {S3_BUCKET}")
    except Exception as e:
        print(f"Error initializing S3 client within audio_api: {e}")
else:
    print("AWS S3 credentials or bucket name not fully set in audio_api. S3 upload/download functionality will be disabled.")


# Helper class to mimic file-like object for boto3.upload_fileobj
class FileContent:
    def __init__(self, content: bytes):
        self._content = content
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            chunk = self._content[self._position:]
            self._position = len(self._content)
            return chunk
        else:
            chunk = self._content[self._position : self._position + size]
            self._position += len(chunk)
            return chunk

    def seek(self, offset: int, whence: int = 0):
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        elif whence == 2:
            self._position = len(self._content) + offset
        else:
            raise ValueError("Invalid whence value")

router = APIRouter()

# Define static phrase cards data in the backend for progress tracking.
# In a more complex application, this data might be stored in a database
# or a separate configuration file to avoid duplication with the frontend.
PHRASE_CARDS_DATA = [
    {
      "id": 'set1',
      "name": 'Common Phrases',
      "phrases": [
        'Hello, how are you?', 'What is your name?', 'Nice to meet you.',
        'Please pass the salt.', 'Thank you very much.', 'Excuse me, where is the restroom?',
        'I need some help.', 'Can you repeat that?', 'I understand now.',
        'How much does this cost?', 'I would like to order.', 'This is delicious.',
        'I am feeling well.', 'What time is it?', 'Could you spell that?',
        'I live in New York.', 'My favorite color is blue.', 'I enjoy reading books.',
        'Do you have any questions?', 'Have a good day!',
      ],
    },
    {
      "id": 'set2',
      "name": 'Daily Activities',
      "phrases": [
        'I am waking up early.', 'I need to brush my teeth.', 'Time to make breakfast.',
        'I am going to work.', 'Driving my car to the store.', 'Cooking dinner tonight.',
        'Washing the dishes.', 'Taking a short nap.', 'Going for a walk.',
        'Reading a newspaper.', 'Watching television.', 'Calling a friend.',
        'Preparing for bed.', 'Turning off the lights.', 'Having a cup of tea.',
        'Walking the dog.', 'Cleaning the house.', 'Doing laundry.',
        'Watering the plants.', 'Listening to music.',
      ],
    },
    {
      "id": 'set3',
      "name": 'Expressing Emotions',
      "phrases": [
        'I am very happy today.', 'I feel a bit sad.', 'This makes me angry.',
        'I am so excited!', 'I am feeling nervous.', 'I am truly grateful.',
        'This is frustrating.', 'I am proud of you.', 'I am feeling anxious.',
        'I am so relieved.', 'This is very confusing.', 'I feel a strong sense of joy.',
        'I am quite surprised.', 'I am feeling disappointed.', 'I am so bored.',
        'I feel truly inspired.', 'I am feeling overwhelmed.', 'This is amazing!',
        'I am sorry.', 'I am content.',
      ],
    },
    {
      "id": 'set4',
      "name": 'Situational Responses',
      "phrases": [
        'Yes, I agree.', 'No, thank you.', 'Maybe next time.', 'I am not sure.',
        'I will think about it.', 'Could you help me, please?', 'I apologize for the inconvenience.',
        'It was my pleasure.', 'I need to leave now.', 'I will be right back.',
        'Please wait for me.', 'I am looking for this item.', 'How can I assist you?',
        'I understand your concern.', 'Let me check for you.', 'Is there anything else?',
        'I am ready to go.', 'Can I have the bill?', 'I would like to pay now.',
        'Take care.',
      ],
    },
]

# Pydantic model for a single phrase set's progress
class PhraseSetProgress(BaseModel):
    set_id: str
    set_name: str
    total_phrases: int
    completed_phrases: int
    is_complete: bool

# Pydantic model for the overall patient progress report
class PatientProgressReport(BaseModel):
    patient_id: str
    patient_name: str
    slp_id: str
    sets_progress: List[PhraseSetProgress]
    overall_completion_percentage: float

# Pydantic model for the POST request body for patient progress
class PatientProgressRequest(BaseModel):
    slp_id: str
    patient_id: str

# Pydantic model for the POST request body for getting patients by SLP ID
class SLPGetPatientsRequest(BaseModel):
    slp_id: str

# NEW Pydantic model for the POST request body for generating a presigned URL
class PresignedUrlRequest(BaseModel):
    slp_id: str
    patient_id: str
    recording_uuid: str # This UUID refers to the 'id' column of the recordings table
    phrase_number: int

router = APIRouter()

# Helper function to create a presigned URL
def create_presigned_url(bucket_name: str, object_name: str, expiration: int = 3600):
    """
    Generate a presigned URL to share an S3 object.
    :param bucket_name: String name of the S3 bucket.
    :param object_name: String name of the S3 object (key).
    :param expiration: Time in seconds for the presigned URL to remain valid.
    :return: String presigned URL.
    """
    if s3_client is None:
        print("S3 client not initialized, cannot create presigned URL.")
        return None

    try:
        response = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=expiration
        )
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return None
    except NoCredentialsError:
        print("AWS credentials not found. Cannot generate presigned URL.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while generating presigned URL: {e}")
        return None
    return response

@router.post("/audio_upload", response_model=MessageResponse) # CHANGED PATH
async def upload_audio(
    file: UploadFile = File(...),
    slp_id: str = Form(...),
    patient_id: str = Form(...),
    phrase_number: int = Form(...),
    phrase_text: str = Form(...),
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Handles audio file uploads to AWS S3 and updates PostgreSQL with recording metadata.
    File name format: slpid_patientid_audio_uuid_phrasenumber.wav
    """
    if s3_client is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="AWS S3 client not initialized. Check credentials.", success=False)
    if S3_BUCKET is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="S3 bucket name not configured.", success=False)
    if AWS_REGION is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="AWS region not configured.", success=False)


    # Generate a UUID for uniqueness for the audio file and its metadata entry
    file_uuid = str(uuid.uuid4())
    s3_key = f"{slp_id}/{patient_id}/recordings/{slp_id}_{patient_id}_audio_{file_uuid}_phrase{phrase_number}.wav"

    try:
        # Read file content
        file_content = await file.read()

        # Upload to S3
        s3_client.upload_fileobj(
            FileContent(file_content),
            S3_BUCKET,
            s3_key,
            ExtraArgs={'ContentType': file.content_type}
        )
        s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

        # Insert recording metadata into PostgreSQL
        # First, verify if patient_id actually exists in the patients table
        patient_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM patients WHERE patient_id = $1 AND slp_id = $2)",
            patient_id, slp_id
        )
        if not patient_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Patient with ID {patient_id} under SLP {slp_id} not found. Cannot store recording metadata.",
                                success=False)

        await conn.execute(
            """
            INSERT INTO recordings (slp_id, patient_id, phrase_number, phrase_text, s3_key, s3_url, content_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            slp_id,
            patient_id,
            phrase_number,
            phrase_text,
            s3_key,
            s3_url,
            file.content_type
        )

        return MessageResponse(message="Audio uploaded and metadata saved successfully!", detail=s3_url, success=True)
    except NoCredentialsError:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="AWS credentials not found. Please configure them.", success=False)
    except ClientError as e:
        print(f"S3 Client Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"S3 upload failed: {e}", success=False)
    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        print(f"Error during audio upload or metadata storage: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to upload audio or save metadata: {e}", success=False)

# MODIFIED: Changed from GET to POST and moved parameters to request body
@router.post("/audio_presigned_url", response_model=MessageResponse)
async def get_presigned_audio_url(
    request_body: PresignedUrlRequest, # Use the new Pydantic model for the request body
    conn: asyncpg.Connection = Depends(get_db_connection)
):
    """
    Generates a presigned URL for a specific audio recording.
    The recording metadata is checked from PostgreSQL.
    Accepts slp_id, patient_id, recording_uuid, and phrase_number in the request body.
    """
    slp_id = request_body.slp_id
    patient_id = request_body.patient_id
    recording_uuid = request_body.recording_uuid
    phrase_number = request_body.phrase_number

    if s3_client is None or S3_BUCKET is None or AWS_REGION is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="S3 configuration missing or client not initialized.", success=False)

    # Retrieve s3_key from PostgreSQL using recording_uuid (id column)
    # Note: recording_uuid in the URL now refers to the UUID from the `id` column of the recordings table.
    # We also need to ensure it belongs to the correct SLP and Patient for security.
    recording_record = await conn.fetchrow(
        "SELECT s3_key FROM recordings WHERE id = $1 AND slp_id = $2 AND patient_id = $3 AND phrase_number = $4",
        recording_uuid, slp_id, patient_id, phrase_number
    )

    if not recording_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Recording metadata not found in database or does not match provided details.", success=False)

    object_name = recording_record["s3_key"] # Use the S3 key retrieved from the database

    presigned_url = create_presigned_url(S3_BUCKET, object_name)

    if presigned_url:
        return MessageResponse(message="Presigned URL generated successfully", detail=presigned_url, success=True)
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to generate presigned URL.", success=False)