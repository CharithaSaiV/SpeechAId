# # api/slp_api.py
# from fastapi import APIRouter, Depends, HTTPException, status
# import datetime
# from pydantic import BaseModel
# from typing import List, Dict, Any, Optional
# import uuid
# import asyncpg

# from models import SLPSignupRequest, PatientRegistrationRequest, MessageResponse
# from services.database import get_db_connection

# router = APIRouter()

# # Define static phrase cards data in the backend for progress tracking.
# # In a more complex application, this data might be stored in a database
# # or a separate configuration file to avoid duplication with the frontend.
# PHRASE_CARDS_DATA = [
#     {
#       "id": 'set1',
#       "name": 'Common Phrases',
#       "phrases": [
#         'Hello, how are you?', 'What is your name?', 'Nice to meet you.',
#         'Please pass the salt.', 'Thank you very much.', 'Excuse me, where is the restroom?',
#         'I need some help.', 'Can you repeat that?', 'I understand now.',
#         'How much does this cost?', 'I would like to order.', 'This is delicious.',
#         'I am feeling well.', 'What time is it?', 'Could you spell that?',
#         'I live in New York.', 'My favorite color is blue.', 'I enjoy reading books.',
#         'Do you have any questions?', 'Have a good day!',
#       ],
#     },
#     {
#       "id": 'set2',
#       "name": 'Daily Activities',
#       "phrases": [
#         'I am waking up early.', 'I need to brush my teeth.', 'Time to make breakfast.',
#         'I am going to work.', 'Driving my car to the store.', 'Cooking dinner tonight.',
#         'Washing the dishes.', 'Taking a short nap.', 'Going for a walk.',
#         'Reading a newspaper.', 'Watching television.', 'Calling a friend.',
#         'Preparing for bed.', 'Turning off the lights.', 'Having a cup of tea.',
#         'Walking the dog.', 'Cleaning the house.', 'Doing laundry.',
#         'Watering the plants.', 'Listening to music.',
#       ],
#     },
#     {
#       "id": 'set3',
#       "name": 'Expressing Emotions',
#       "phrases": [
#         'I am very happy today.', 'I feel a bit sad.', 'This makes me angry.',
#         'I am so excited!', 'I am feeling nervous.', 'I am truly grateful.',
#         'This is frustrating.', 'I am proud of you.', 'I am feeling anxious.',
#         'I am so relieved.', 'This is very confusing.', 'I feel a strong sense of joy.',
#         'I am quite surprised.', 'I am feeling disappointed.', 'I am so bored.',
#         'I feel truly inspired.', 'I am feeling overwhelmed.', 'This is amazing!',
#         'I am sorry.', 'I am content.',
#       ],
#     },
#     {
#       "id": 'set4',
#       "name": 'Situational Responses',
#       "phrases": [
#         'Yes, I agree.', 'No, thank you.', 'Maybe next time.', 'I am not sure.',
#         'I will think about it.', 'Could you help me, please?', 'I apologize for the inconvenience.',
#         'It was my pleasure.', 'I need to leave now.', 'I will be right back.',
#         'Please wait for me.', 'I am looking for this item.', 'How can I assist you?',
#         'I understand your concern.', 'Let me check for you.', 'Is there anything else?',
#         'I am ready to go.', 'Can I have the bill?', 'I would like to pay now.',
#         'Take care.',
#       ],
#     },
# ]

# # Pydantic model for a single phrase set's progress
# class PhraseSetProgress(BaseModel):
#     set_id: str
#     set_name: str
#     total_phrases: int
#     completed_phrases: int
#     is_complete: bool

# # Pydantic model for the overall patient progress report
# class PatientProgressReport(BaseModel):
#     patient_id: str
#     patient_name: str
#     slp_id: str
#     sets_progress: List[PhraseSetProgress]
#     total_progress: float

# # Pydantic model for the POST request body for patient progress
# class PatientProgressRequest(BaseModel):
#     slp_id: str
#     patient_id: str

# # NEW Pydantic model for the POST request body for getting patients by SLP ID
# class SLPGetPatientsRequest(BaseModel):
#     slp_id: str

# @router.post("/signup", response_model=MessageResponse)
# async def signup(slp_details: SLPSignupRequest, conn: asyncpg.Connection = Depends(get_db_connection)):
#     """
#     Registers a new Speech-Language Pathologist (SLP) in PostgreSQL.
#     If an SLP with the given email already exists, it's treated as a login.
#     Otherwise, a new SLP is registered.
#     """
#     try:
#         # 1. Try to find SLP by email (primary check for "login")
#         existing_slp_by_email = await conn.fetchrow(
#             "SELECT slp_id FROM slps WHERE email = $1",
#             slp_details.email
#         )
#         if existing_slp_by_email:
#             # SLP found by email, treat as login
#             return MessageResponse(
#                 message="SLP logged in successfully!",
#                 detail=f"SLP ID: {existing_slp_by_email['slp_id']}", # Return the existing SLP ID from DB
#                 success=True
#             )

#         # 2. If no SLP found by email, check if SLP_ID is already taken (conflict for new registration)
#         # This prevents a new email from being registered with an SLP_ID that's already in use
#         # by a different email.
#         existing_slp_by_id = await conn.fetchrow(
#             "SELECT slp_id FROM slps WHERE slp_id = $1",
#             slp_details.slp_id
#         )
#         if existing_slp_by_id:
#             raise HTTPException(
#                 status_code=status.HTTP_409_CONFLICT,
#                 detail=f"SLP ID '{slp_details.slp_id}' is already registered with a different email.",
#                 headers={"X-Error-Reason": "Duplicate SLP ID"}
#             )

#         # 3. If neither email nor SLP_ID exists, this is a truly new registration
#         await conn.execute(
#             "INSERT INTO slps (slp_id, name, email) VALUES ($1, $2, $3)",
#             slp_details.slp_id, slp_details.name, slp_details.email
#         )
#         return MessageResponse(
#             message="SLP registered successfully!",
#             detail=f"SLP ID: {slp_details.slp_id}",
#             success=True
#         )
#     except HTTPException as e:
#         # Re-raise HTTPExceptions, they will be caught by the global exception handler
#         raise e
#     except Exception as e:
#         print(f"Error during SLP signup/login: {e}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process SLP request: {e}")

# @router.post("/register_patient", response_model=MessageResponse)
# async def register_patient(patient_details: PatientRegistrationRequest, conn: asyncpg.Connection = Depends(get_db_connection)):
#     """
#     Registers a new patient under a specific SLP in PostgreSQL.
#     If a patient with the same name and clinic_name already exists under this SLP,
#     returns the existing patient's ID. Otherwise, registers a new patient.
#     """
#     try:
#         # Ensure the SLP exists
#         slp_exists = await conn.fetchval(
#             "SELECT EXISTS(SELECT 1 FROM slps WHERE slp_id = $1)",
#             patient_details.slp_id
#         )
#         if not slp_exists:
#             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
#                                 detail=f"SLP with ID {patient_details.slp_id} not found. Please register the SLP first.",
#                                 headers={"X-Error-Reason": "SLP Not Found"})

#         # Check if patient with same name and clinic_name already exists under this SLP
#         existing_patient = await conn.fetchrow(
#             "SELECT patient_id FROM patients WHERE slp_id = $1 AND name = $2 AND clinic_name = $3",
#             patient_details.slp_id, patient_details.name, patient_details.clinic_name
#         )

#         if existing_patient:
#             # Patient already exists under this SLP with the same name and clinic_name
#             return MessageResponse(
#                 message="Patient already registered under this SLP.",
#                 detail=f"Patient ID: {existing_patient['patient_id']}",
#                 success=True
#             )

#         # If not found, generate a unique patient_id and insert new patient
#         generated_patient_id = str(uuid.uuid4())

#         await conn.execute(
#             """
#             INSERT INTO patients (patient_id, name, age, ailment, severity, gender, clinic_name, slp_id)
#             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
#             """,
#             generated_patient_id,
#             patient_details.name,
#             patient_details.age,
#             patient_details.ailment,
#             patient_details.severity,
#             patient_details.gender,
#             patient_details.clinic_name,
#             patient_details.slp_id
#         )

#         return MessageResponse(message="Patient registered successfully!", detail=f"Patient ID: {generated_patient_id}", success=True)
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         print(f"Error during patient registration: {e}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to register patient: {e}")

# # MODIFIED: Changed from GET to POST and moved slp_id to request body
# # ADDED: Overall progress for each patient
# @router.post("/get_patients
#", response_model=List[Dict[str, Any]])
# async def get_patients_for_slp(request_body: SLPGetPatientsRequest, conn: asyncpg.Connection = Depends(get_db_connection)):
#     """
#     Retrieves all patients registered under a given SLP from PostgreSQL,
#     including their overall progress percentage.
#     Accepts slp_id in the request body.
#     """
#     slp_id = request_body.slp_id
#     try:
#         patients_records = await conn.fetch(
#             "SELECT id, patient_id, name, age, ailment, severity, gender, clinic_name, slp_id, registered_at FROM patients WHERE slp_id = $1",
#             slp_id
#         )
#         patients_list = []
#         for patient_record in patients_records:
#             patient_data = dict(patient_record) # Convert Row to dict

#             # Calculate progress for this patient (logic adapted from get_patient_progress)
#             recorded_phrases_data = await conn.fetch(
#                 "SELECT phrase_text, phrase_number FROM recordings WHERE slp_id = $1 AND patient_id = $2",
#                 slp_id, patient_data['patient_id']
#             )
#             recorded_phrases_set = {(r['phrase_text'], r['phrase_number']) for r in recorded_phrases_data}

#             total_recorded_phrases_overall = 0
#             total_possible_phrases_overall = 0

#             for phrase_set in PHRASE_CARDS_DATA:
#                 total_phrases_in_set = len(phrase_set["phrases"])
#                 completed_phrases_in_set = 0
#                 for i, phrase_text in enumerate(phrase_set["phrases"]):
#                     phrase_number = i + 1
#                     if (phrase_text, phrase_number) in recorded_phrases_set:
#                         completed_phrases_in_set += 1
#                 total_recorded_phrases_overall += completed_phrases_in_set
#                 total_possible_phrases_overall += total_phrases_in_set
            
#             total_progress = 0.0
#             if total_possible_phrases_overall > 0:
#                 total_progress = (total_recorded_phrases_overall / total_possible_phrases_overall) * 100
            
#             patient_data['total_progress'] = round(total_progress, 2)
#             patients_list.append(patient_data)

#         return patients_list
#     except Exception as e:
#         print(f"Error fetching patients for SLP {slp_id}: {e}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve patients: {e}")


# # MODIFIED: Changed from GET to POST and moved parameters to request body
# @router.post("/get_patient_progress", response_model=PatientProgressReport)
# async def get_patient_progress(request_body: PatientProgressRequest, conn: asyncpg.Connection = Depends(get_db_connection)):
#     """
#     Tracks and returns the progress of a specific patient across all phrase sets.
#     Reports which sets are completed and how many phrases within each set are finished.
#     Accepts slp_id and patient_id in the request body.
#     """
#     slp_id = request_body.slp_id
#     patient_id = request_body.patient_id

#     try:
#         # 1. Verify SLP and Patient existence
#         patient_record = await conn.fetchrow(
#             "SELECT name FROM patients WHERE slp_id = $1 AND patient_id = $2",
#             slp_id, patient_id
#         )
#         if not patient_record:
#             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
#                                 detail=f"Patient with ID {patient_id} not found under SLP {slp_id}.")

#         patient_name = patient_record['name']

#         # 2. Fetch all recordings for this patient
#         recorded_phrases_data = await conn.fetch(
#             "SELECT phrase_text, phrase_number FROM recordings WHERE slp_id = $1 AND patient_id = $2",
#             slp_id, patient_id
#         )

#         # Create a set of recorded phrases for quick lookup: (phrase_text, phrase_number)
#         recorded_phrases_set = {(r['phrase_text'], r['phrase_number']) for r in recorded_phrases_data}

#         sets_progress: List[PhraseSetProgress] = []
#         total_recorded_phrases_overall = 0
#         total_possible_phrases_overall = 0

#         # 3. Iterate through predefined phrase sets and calculate progress
#         for phrase_set in PHRASE_CARDS_DATA:
#             set_id = phrase_set["id"]
#             set_name = phrase_set["name"]
#             total_phrases_in_set = len(phrase_set["phrases"])
#             completed_phrases_in_set = 0

#             for i, phrase_text in enumerate(phrase_set["phrases"]):
#                 phrase_number = i + 1 # Phrase numbers are 1-indexed
#                 if (phrase_text, phrase_number) in recorded_phrases_set:
#                     completed_phrases_in_set += 1
            
#             is_set_complete = (completed_phrases_in_set == total_phrases_in_set)

#             sets_progress.append(PhraseSetProgress(
#                 set_id=set_id,
#                 set_name=set_name,
#                 total_phrases=total_phrases_in_set,
#                 completed_phrases=completed_phrases_in_set,
#                 is_complete=is_set_complete
#             ))
#             total_recorded_phrases_overall += completed_phrases_in_set
#             total_possible_phrases_overall += total_phrases_in_set
        
#         total_progress = 0.0
#         if total_possible_phrases_overall > 0:
#             total_progress = (total_recorded_phrases_overall / total_possible_phrases_overall) * 100

#         return PatientProgressReport(
#             patient_id=patient_id,
#             patient_name=patient_name,
#             slp_id=slp_id,
#             sets_progress=sets_progress,
#             total_progress=round(total_progress, 2)
#         )

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         print(f"Error fetching patient progress for SLP {slp_id}, Patient {patient_id}: {e}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve patient progress: {e}")

# api/slp_api.py
from fastapi import APIRouter, Depends, HTTPException, status
import datetime
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import asyncpg

from models import SLPSignupRequest, PatientRegistrationRequest, MessageResponse
from services.database import get_db_connection


router = APIRouter()

# Define static phrase cards data in the backend as the source of truth.
# UPDATED: Structure to include phrases_id and phrases_name
PHRASE_CARDS_DATA = [
    {
      "id": 'set1',
      "name": 'Basic Greetings',
      "phrases": [
        {"phrases_id": "1", "phrases_name": "Hello, how are you today?"},
        {"phrases_id": "2", "phrases_name": "Good morning, have a great day!"},
        {"phrases_id": "3", "phrases_name": "Thank you very much for your help."},
        {"phrases_id": "4", "phrases_name": "Please let me know if you need anything."},
        {"phrases_id": "5", "phrases_name": "I hope you have a wonderful evening."},
        {"phrases_id": "6", "phrases_name": "It was nice meeting you today."},
        {"phrases_id": "7", "phrases_name": "Have a safe trip home."},
        {"phrases_id": "8", "phrases_name": "See you again soon."},
        {"phrases_id": "9", "phrases_name": "Take care of yourself."},
        {"phrases_id": "10", "phrases_name": "Good luck with your presentation."},
        {"phrases_id": "11", "phrases_name": "Congratulations on your achievement."},
        {"phrases_id": "12", "phrases_name": "I appreciate your time and effort."},
        {"phrases_id": "13", "phrases_name": "Welcome to our community."},
        {"phrases_id": "14", "phrases_name": "Please feel free to ask questions."},
        {"phrases_id": "15", "phrases_name": "I look forward to hearing from you."},
        {"phrases_id": "16", "phrases_name": "Thank you for your patience."},
        {"phrases_id": "17", "phrases_name": "Have a blessed day ahead."},
        {"phrases_id": "18", "phrases_name": "It was a pleasure working with you."},
        {"phrases_id": "19", "phrases_name": "Best wishes for your future endeavors."},
        {"phrases_id": "20", "phrases_name": "May you find success in everything you do."},
      ],
    },
    {
      "id": 'set2',
      "name": 'Business Communication',
      "phrases": [
        {"phrases_id": "1", "phrases_name": "I would like to schedule a meeting."},
        {"phrases_id": "2", "phrases_name": "Could you please send me the report?"},
        {"phrases_id": "3", "phrases_name": "Let me get back to you on this."},
        {"phrases_id": "4", "phrases_name": "I need to review the documents first."},
        {"phrases_id": "5", "phrases_name": "The deadline for this project is Friday."},
        {"phrases_id": "6", "phrases_name": "We should discuss this in more detail."},
        {"phrases_id": "7", "phrases_name": "I will forward your message to the team."},
        {"phrases_id": "8", "phrases_name": "Please confirm your availability."},
        {"phrases_id": "9", "phrases_name": "The meeting has been rescheduled."},
        {"phrases_id": "10", "phrases_name": "I apologize for the inconvenience."},
        {"phrases_id": "11", "phrases_name": "We need to finalize the budget."},
        {"phrases_id": "12", "phrases_name": "I'll prepare the presentation for next week."},
        {"phrases_id": "13", "phrases_name": "Let's align on the strategy moving forward."},
        {"phrases_id": "14", "phrases_name": "I'll send out the meeting minutes shortly."},
        {"phrases_id": "15", "phrases_name": "We need to ensure compliance with regulations."},
        {"phrases_id": "16", "phrases_name": "The project is progressing as planned."},
        {"phrases_id": "17", "phrases_name": "Please provide your feedback by end of day."},
        {"phrases_id": "18", "phrases_name": "I'm looking forward to your insights."},
        {"phrases_id": "19", "phrases_name": "Let's prioritize these tasks."},
        {"phrases_id": "20", "phrases_name": "Thank you for your prompt response."},
      ],
    },
    {
      "id": 'set3',
      "name": 'Expressing Emotions',
      "phrases": [
        {"phrases_id": "1", "phrases_name": "I am very happy today."},
        {"phrases_id": "2", "phrases_name": "I feel a bit sad."},
        {"phrases_id": "3", "phrases_name": "This makes me angry."},
        {"phrases_id": "4", "phrases_name": "I am so excited!"},
        {"phrases_id": "5", "phrases_name": "I am feeling nervous."},
        {"phrases_id": "6", "phrases_name": "I am truly grateful."},
        {"phrases_id": "7", "phrases_name": "This is frustrating."},
        {"phrases_id": "8", "phrases_name": "I am proud of you."},
        {"phrases_id": "9", "phrases_name": "I am feeling anxious."},
        {"phrases_id": "10", "phrases_name": "I am so relieved."},
        {"phrases_id": "11", "phrases_name": "This is very confusing."},
        {"phrases_id": "12", "phrases_name": "I feel a strong sense of joy."},
        {"phrases_id": "13", "phrases_name": "I am quite surprised."},
        {"phrases_id": "14", "phrases_name": "I am feeling disappointed."},
        {"phrases_id": "15", "phrases_name": "I am so bored."},
        {"phrases_id": "16", "phrases_name": "I feel truly inspired."},
        {"phrases_id": "17", "phrases_name": "I am feeling overwhelmed."},
        {"phrases_id": "18", "phrases_name": "This is amazing!"},
        {"phrases_id": "19", "phrases_name": "I am sorry."},
        {"phrases_id": "20", "phrases_name": "I am content."},
      ],
    },
    {
      "id": 'set4',
      "name": 'Situational Responses',
      "phrases": [
        {"phrases_id": "1", "phrases_name": "Yes, I agree."},
        {"phrases_id": "2", "phrases_name": "No, thank you."},
        {"phrases_id": "3", "phrases_name": "Maybe next time."},
        {"phrases_id": "4", "phrases_name": "I am not sure."},
        {"phrases_id": "5", "phrases_name": "I will think about it."},
        {"phrases_id": "6", "phrases_name": "Could you help me, please?"},
        {"phrases_id": "7", "phrases_name": "I apologize for the inconvenience."},
        {"phrases_id": "8", "phrases_name": "It was my pleasure."},
        {"phrases_id": "9", "phrases_name": "I need to leave now."},
        {"phrases_id": "10", "phrases_name": "I will be right back."},
        {"phrases_id": "11", "phrases_name": "Please wait for me."},
        {"phrases_id": "12", "phrases_name": "I am looking for this item."},
        {"phrases_id": "13", "phrases_name": "How can I assist you?"},
        {"phrases_id": "14", "phrases_name": "I understand your concern."},
        {"phrases_id": "15", "phrases_name": "Let me check for you."},
        {"phrases_id": "16", "phrases_name": "Is there anything else?"},
        {"phrases_id": "17", "phrases_name": "I am ready to go."},
        {"phrases_id": "18", "phrases_name": "Can I have the bill?"},
        {"phrases_id": "19", "phrases_name": "I would like to pay now."},
        {"phrases_id": "20", "phrases_name": "Take care."},
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
    total_progress: float

# Pydantic model for the POST request body for patient progress
class PatientProgressRequest(BaseModel):
    slp_id: str
    patient_id: str

# Pydantic model for the POST request body for getting patients by SLP ID
class SLPGetPatientsRequest(BaseModel):
    slp_id: str

# Pydantic model for the POST request body for generating a presigned URL
class PresignedUrlRequest(BaseModel):
    slp_id: str
    patient_id: str
    recording_uuid: str # This UUID refers to the 'id' column of the recordings table
    phrase_number: int

# NEW: Pydantic model for the response when getting phrase set names
class PhraseSetNameResponse(BaseModel):
    id: str
    name: str

# NEW: Pydantic model for a single phrase within a set
class PhraseDetail(BaseModel):
    phrases_id: str
    phrases_name: str

# NEW: Pydantic model for the request body when getting phrases for a specific set
class GetPhraseSetPhrasesRequest(BaseModel):
    set_id: str

# NEW: Pydantic model for the response when getting phrases for a specific set
class PhraseSetPhrasesResponse(BaseModel):
    id: str
    name: str
    phrases: List[PhraseDetail] # List of PhraseDetail objects

@router.post("/signup", response_model=MessageResponse)
async def signup(slp_details: SLPSignupRequest, conn: asyncpg.Connection = Depends(get_db_connection)):
    """
    Registers a new Speech-Language Pathologist (SLP) in PostgreSQL.
    If an SLP with the given email already exists, it's treated as a login.
    Otherwise, a new SLP is registered.
    """
    try:
        # 1. Try to find SLP by email (primary check for "login")
        existing_slp_by_email = await conn.fetchrow(
            "SELECT slp_id FROM slps WHERE email = $1",
            slp_details.email
        )
        if existing_slp_by_email:
            # SLP found by email, treat as login
            return MessageResponse(
                message="SLP logged in successfully!",
                detail=f"SLP ID: {existing_slp_by_email['slp_id']}", # Return the existing SLP ID from DB
                success=True
            )

        # 2. If no SLP found by email, check if SLP_ID is already taken (conflict for new registration)
        # This prevents a new email from being registered with an SLP_ID that's already in use
        # by a different email.
        existing_slp_by_id = await conn.fetchrow(
            "SELECT slp_id FROM slps WHERE slp_id = $1",
            slp_details.slp_id
        )
        if existing_slp_by_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"SLP ID '{slp_details.slp_id}' is already registered with a different email.",
                headers={"X-Error-Reason": "Duplicate SLP ID"}
            )

        # 3. If neither email nor SLP_ID exists, this is a truly new registration
        await conn.execute(
            "INSERT INTO slps (slp_id, name, email) VALUES ($1, $2, $3)",
            slp_details.slp_id, slp_details.name, slp_details.email
        )
        return MessageResponse(
            message="SLP registered successfully!",
            detail=f"SLP ID: {slp_details.slp_id}",
            success=True
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions, they will be caught by the global exception handler
        raise e
    except Exception as e:
        print(f"Error during SLP signup/login: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process SLP request: {e}")

@router.post("/register_patient", response_model=MessageResponse)
async def register_patient(patient_details: PatientRegistrationRequest, conn: asyncpg.Connection = Depends(get_db_connection)):
    """
    Registers a new patient under a specific SLP in PostgreSQL.
    If a patient with the same name and clinic_name already exists under this SLP,
    returns the existing patient's ID. Otherwise, registers a new patient.
    """
    try:
        # Ensure the SLP exists
        slp_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM slps WHERE slp_id = $1)",
            patient_details.slp_id
        )
        if not slp_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"SLP with ID {patient_details.slp_id} not found. Please register the SLP first.",
                                headers={"X-Error-Reason": "SLP Not Found"})

        # Check if patient with same name and clinic_name already exists under this SLP
        existing_patient = await conn.fetchrow(
            "SELECT patient_id FROM patients WHERE slp_id = $1 AND name = $2 AND clinic_name = $3",
            patient_details.slp_id, patient_details.name, patient_details.clinic_name
        )

        if existing_patient:
            # Patient already exists under this SLP with the same name and clinic_name
            return MessageResponse(
                message="Patient already registered under this SLP.",
                detail=f"Patient ID: {existing_patient['patient_id']}",
                success=True
            )

        # If not found, generate a unique patient_id and insert new patient
        generated_patient_id = str(uuid.uuid4())

        await conn.execute(
            """
            INSERT INTO patients (patient_id, name, age, ailment, severity, gender, clinic_name, slp_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            generated_patient_id,
            patient_details.name,
            patient_details.age,
            patient_details.ailment,
            patient_details.severity,
            patient_details.gender,
            patient_details.clinic_name,
            patient_details.slp_id
        )

        return MessageResponse(message="Patient registered successfully!", detail=f"Patient ID: {generated_patient_id}", success=True)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during patient registration: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to register patient: {e}")

# MODIFIED: Changed from GET to POST and moved slp_id to request body
# ADDED: Overall progress for each patient
@router.post("/get_patients", response_model=List[Dict[str, Any]])
async def get_patients_for_slp(request_body: SLPGetPatientsRequest, conn: asyncpg.Connection = Depends(get_db_connection)):
    """
    Retrieves all patients registered under a given SLP from PostgreSQL,
    including their overall progress percentage.
    Accepts slp_id in the request body.
    """
    slp_id = request_body.slp_id
    try:
        patients_records = await conn.fetch(
            "SELECT id, patient_id, name, age, ailment, severity, gender, clinic_name, slp_id, registered_at FROM patients WHERE slp_id = $1",
            slp_id
        )
        patients_list = []
        for patient_record in patients_records:
            patient_data = dict(patient_record) # Convert Row to dict

            # Calculate progress for this patient (logic adapted from get_patient_progress)
            recorded_phrases_data = await conn.fetch(
                "SELECT phrase_text, phrase_number FROM recordings WHERE slp_id = $1 AND patient_id = $2",
                slp_id, patient_data['patient_id']
            )
            recorded_phrases_set = {(r['phrase_text'], r['phrase_number']) for r in recorded_phrases_data}

            total_recorded_phrases_overall = 0
            total_possible_phrases_overall = 0

            for phrase_set in PHRASE_CARDS_DATA:
                total_phrases_in_set = len(phrase_set["phrases"])
                completed_phrases_in_set = 0
                # Note: PHRASE_CARDS_DATA phrases now have 'phrases_name' not just string
                for i, phrase_detail in enumerate(phrase_set["phrases"]):
                    phrase_text = phrase_detail["phrases_name"]
                    phrase_number = i + 1 # Phrase numbers are 1-indexed based on list position
                    if (phrase_text, phrase_number) in recorded_phrases_set:
                        completed_phrases_in_set += 1
                total_recorded_phrases_overall += completed_phrases_in_set
                total_possible_phrases_overall += total_phrases_in_set
            
            total_progress = 0.0
            if total_possible_phrases_overall > 0:
                total_progress = (total_recorded_phrases_overall / total_possible_phrases_overall) * 100
            
            patient_data['total_progress'] = round(total_progress, 2)
            patients_list.append(patient_data)

        return patients_list
    except Exception as e:
        print(f"Error fetching patients for SLP {slp_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve patients: {e}")

# MODIFIED: Changed from GET to POST and moved parameters to request body
@router.post("/get_patient_progress", response_model=PatientProgressReport)
async def get_patient_progress(request_body: PatientProgressRequest, conn: asyncpg.Connection = Depends(get_db_connection)):
    """
    Tracks and returns the progress of a specific patient across all phrase sets.
    Reports which sets are completed and how many phrases within each set are finished.
    Accepts slp_id and patient_id in the request body.
    """
    slp_id = request_body.slp_id
    patient_id = request_body.patient_id

    try:
        # 1. Verify SLP and Patient existence
        patient_record = await conn.fetchrow(
            "SELECT name FROM patients WHERE slp_id = $1 AND patient_id = $2",
            slp_id, patient_id
        )
        if not patient_record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Patient with ID {patient_id} not found under SLP {slp_id}.")

        patient_name = patient_record['name']

        # 2. Fetch all recordings for this patient
        recorded_phrases_data = await conn.fetch(
            "SELECT phrase_text, phrase_number FROM recordings WHERE slp_id = $1 AND patient_id = $2",
            slp_id, patient_id
        )

        # Create a set of recorded phrases for quick lookup: (phrase_text, phrase_number)
        recorded_phrases_set = {(r['phrase_text'], r['phrase_number']) for r in recorded_phrases_data}

        sets_progress: List[PhraseSetProgress] = []
        total_recorded_phrases_overall = 0
        total_possible_phrases_overall = 0

        # 3. Iterate through predefined phrase sets and calculate progress
        for phrase_set in PHRASE_CARDS_DATA:
            set_id = phrase_set["id"]
            set_name = phrase_set["name"]
            total_phrases_in_set = len(phrase_set["phrases"])
            completed_phrases_in_set = 0

            # Note: PHRASE_CARDS_DATA phrases now have 'phrases_name' not just string
            for i, phrase_detail in enumerate(phrase_set["phrases"]):
                phrase_text = phrase_detail["phrases_name"]
                phrase_number = i + 1 # Phrase numbers are 1-indexed based on list position
                if (phrase_text, phrase_number) in recorded_phrases_set:
                    completed_phrases_in_set += 1
            
            is_set_complete = (completed_phrases_in_set == total_phrases_in_set)

            sets_progress.append(PhraseSetProgress(
                set_id=set_id,
                set_name=set_name,
                total_phrases=total_phrases_in_set,
                completed_phrases=completed_phrases_in_set,
                is_complete=is_set_complete
            ))
            total_recorded_phrases_overall += completed_phrases_in_set
            total_possible_phrases_overall += total_phrases_in_set
        
        total_progress = 0.0
        if total_possible_phrases_overall > 0:
            total_progress = (total_recorded_phrases_overall / total_possible_phrases_overall) * 100

        return PatientProgressReport(
            patient_id=patient_id,
            patient_name=patient_name,
            slp_id=slp_id,
            sets_progress=sets_progress,
            total_progress=round(total_progress, 2)
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error fetching patient progress for SLP {slp_id}, Patient {patient_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve patient progress: {e}")



# NEW API: Get all phrases for a specific set
@router.post("/get_phrases", response_model=PhraseSetPhrasesResponse)
async def get_phrases(request_body: GetPhraseSetPhrasesRequest):
    """
    Retrieves the full list of phrases (with IDs and names) for a given phrase set ID.
    """
    set_id = request_body.set_id
    for phrase_set in PHRASE_CARDS_DATA:
        if phrase_set["id"] == set_id:
            # Ensure the phrases are returned in the PhraseDetail format
            phrases_detail = [PhraseDetail(phrases_id=p["phrases_id"], phrases_name=p["phrases_name"]) for p in phrase_set["phrases"]]
            return PhraseSetPhrasesResponse(id=phrase_set["id"], name=phrase_set["name"], phrases=phrases_detail)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Phrase set with ID '{set_id}' not found.")

