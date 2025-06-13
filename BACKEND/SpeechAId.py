########################################## Imports #########################################################
import os
import random
import json
import tempfile
import time
import psutil
import warnings
import io
import numpy as np
import re
import subprocess
import traceback
import noisereduce as nr
import librosa.effects

import sys
import datetime
from functools import wraps
from io import StringIO
from contextlib import redirect_stdout

import whisper
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from TTS.api import TTS
from pydub import AudioSegment

from utils.user_storage import store_user, load_user_data

from dotenv import load_dotenv

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import librosa
from tqdm import tqdm

from datasets import load_dataset
import soundfile as sf
from evaluate import load

import nltk
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams

import language_tool_python

#from kokoro import KPipeline

###############################################################################################################
########################### Initializing flask app, setting CORS ##############################################

app = Flask(__name__, static_folder="./build", static_url_path="/")
# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://localhost:3001/","http://localhost:3000/","http://38.188.108.234:5020","http://192.168.0.49:5020","https://speechaid.convogene.ai:5020/","http://38.188.108.234","http://192.168.0.49:5020","https://speechaid.convogene.ai:5020/api/store_user"],
#         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"],
#         "supports_credentials": True
#      }
#     })
CORS(app, resources={r"/*": {"origins": "*"}}) 
socketio = SocketIO(app, cors_allowed_origins="*")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

##############################################################################################################
############################### NLTK post processing of the output Transcription #############################

# Step 1: Remove Repeated Phrases
def remove_repeated_phrases(text, n=1):
    tokens = text.split()
    seen = set()
    cleaned_tokens = []
    for i in range(len(tokens)):
        gram = tuple(tokens[i:i+n])  # Generate n-grams
        if gram not in seen:
            cleaned_tokens.append(tokens[i])
            seen.add(gram)
    return ' '.join(cleaned_tokens)
 
# Step 2: Correct Grammar and Spelling
tool = language_tool_python.LanguageTool('en-US')
 
def correct_grammar(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text
 
# Step 3: Remove Filler Words
def remove_filler_words(text, fillers=["uh", "um"]):
    tokens = text.split()
    return ' '.join([word for word in tokens if word.lower() not in fillers])

##########################################################################################################
####################### Load environment variables and Configure S3 Bucket ###############################

load_dotenv()

AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')

s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

print(f"Connected to S3 bucket: {S3_BUCKET}")

##########################################################################################################
########################### Global Settings and available models #########################################

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Define available models with `type` differentiation
SPEAKERS_DIR = "speakers"
AVAILABLE_VOICES = [{
    "label": "GR",
    "id": "speakers/GR.wav",
    "text": "speakers/GR.txt"
  }]
current_voice_path = AVAILABLE_VOICES[0]["id"]

# Ensure the directory exists
os.makedirs(SPEAKERS_DIR, exist_ok=True)

AVAILABLE_MODELS = [
    {"id": "openai/whisper-small", "label": "Whisper Small Base", "type": "whisper"},
    {"id": "facebook/wav2vec2-large-960h", "label": "Wav2Vec2 Large", "type": "wav2vec2"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/finetuned_m01", "label": "Whisper Fine-tuned on M01", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/Synthetic_Phrasecards", "label": "Whisper Fine-tuned on Phrasecards and Synthetic data", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/Synthetic_Finetuned_V2", "label": "Whisper Fine-tuned on Synthetic data V2", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/GR_V4/", "label": "Whisper Fine-tuned on GR_V4", "type": "whisper"}   
]


##########################################################################################################
########################### Model loading and initializing the base model ################################

# Initial model path
current_model_path = "./Model/finetuned_phrasecards"
current_model_type = "whisper"  # Default model type

# Functions for Whisper and Wav2Vec2 model loading
def load_whisper_model(model_path):
    global model, processor, pipe
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained("openai/whisper-small")

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            torch_dtype=torch_dtype,
            device=0 if device == "cuda" else -1,
            return_timestamps=True
        )
        return True, "Whisper model loaded successfully"
    except Exception as e:
        return False, str(e)

def load_wav2vec2_model(model_path):
    global model, processor, asr_pipeline
    try:
        model = Wav2Vec2ForCTC.from_pretrained(
            model_path,
            torch_dtype=torch_dtype
        )
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model.to(device)

        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if device == "cuda" else -1
        )
        return True, "Wav2Vec2 model loaded successfully"
    except Exception as e:
        return False, str(e)

# Generalized model loader
def load_model(model_path, model_type):
    if model_type == "whisper":
        return load_whisper_model(model_path)
    elif model_type == "wav2vec2":
        return load_wav2vec2_model(model_path)
    else:
        return False, f"Unsupported model type: {model_type}"

# Initial model load
success, message = load_model(current_model_path, current_model_type)
if not success:
    print(f"Failed to load initial model: {message}")
else:
    print(f"{current_model_type.capitalize()} model loaded successfully. Device: {device}")

##########################################################################################################
############################################# TTS Model loading  #########################################

# Load TTS model
# def load_tts_model():
#     """
#     Load the Kokoro TTS model.
#     Returns:
#         KPipeline: Initialized Kokoro TTS pipeline
#     """
#     print("Loading Kokoro TTS model...")
#     try:
#         # Initialize Kokoro TTS pipeline
#         # You can customize the language code and other parameters as needed
#         pipeline = KPipeline(lang_code='a')
#         return pipeline
#     except Exception as e:
#         print(f"Error loading Kokoro TTS model: {str(e)}")
#         traceback.print_exc()
#         return None
# synthesiser = KPipeline(lang_code='a')

##########################################################################################################
############################################# Converting the audio into .wav  #########################################

def convert_to_wav(input_file_path):
    audio = AudioSegment.from_file(input_file_path)
    wav_path = input_file_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

##########################################################################################################
############################################# Initializing S3 client #####################################

# Initialize S3 client
s3_client = boto3.client('s3')

def create_presigned_url(bucket_name, object_name, expiration=3600):

    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name, 'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
    return response

##########################################################################################################
############################################# Uploading to S3, Deleting from s3  #########################################

# Function to upload a S3 file
def upload_to_s3(file_path, filename, bucket_name):
    """Helper function to upload files to S3."""
    s3_key = f"audio/{filename}"
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        print(f"File uploaded successfully: {s3_url}")
        return s3_url
    except FileNotFoundError:
        print(f"The file was not found: {file_path}")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return None
    
# Function to delete a S3 file   
def delete_from_s3(bucket_name, filenames):
    """Helper function to delete multiple files from S3 using only file names."""
    s3_client = boto3.client('s3')
    # Construct the S3 keys from filenames
    objects_to_delete = [{'Key': f'audio/{filename}'} for filename in filenames]
    try:
        response = s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={
                'Objects': objects_to_delete
            }
        )   
        # Check if any files were successfully deleted
        deleted_files = response.get('Deleted', [])
        if deleted_files:
            print(f"Deleted files: {[obj['Key'] for obj in deleted_files]}")
        else:
            print("No files were deleted.")
        # Check for errors
        if 'Errors' in response:
            for error in response['Errors']:
                print(f"Error deleting {error['Key']}: {error['Message']}")
    except NoCredentialsError:
        print("Credentials not available")
    except ClientError as e:
        print(f"Error deleting from S3: {str(e)}")

# Function to rename an S3 file
def rename_s3_file(bucket_name, old_filename, new_filename):
    """Helper function to rename a file in S3 by copying it and then deleting the old one."""
    try:
        copy_source = {'Bucket': bucket_name, 'Key': f"audio/{old_filename}"}
        s3_client.copy_object(Bucket=bucket_name, CopySource=copy_source, Key=f"audio/{new_filename}")
        s3_client.delete_object(Bucket=bucket_name, Key=f"audio/{old_filename}")
        new_s3_url = f"https://{bucket_name}.s3.amazonaws.com/audio/{new_filename}"
        return new_s3_url
    except Exception as e:
        print(f"Error renaming file: {str(e)}")
        return None

##########################################################################################################
############################################# calculating word accuracy  #########################################

def calculate_word_accuracy(reference, prediction):
    """Calculate word accuracy between reference and prediction texts."""
    wer_metric = load("wer")
    normalizer = BasicTextNormalizer()
    normalized_reference = normalizer(reference)
    normalized_prediction = normalizer(prediction)
    wer = wer_metric.compute(references=[normalized_reference], predictions=[normalized_prediction])
    word_accuracy = (1 - wer) * 100
    return word_accuracy

##########################################################################################################
############################################# Saving metadat to json  ####################################

def save_metadata(metadata, filename="metadata.json"):
    """Helper function to save metadata to a single JSON file on the server."""
    metadata_filepath = os.path.join(os.getcwd(), filename)
    
    try:
        # Load existing data or create empty list
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, 'r') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Ensure metadata has required fields
        metadata.update({
            "id": metadata.get("id") or f"{random.randint(1000, 9999)}_{int(time.time())}",
            "timestamp": metadata.get("timestamp") or int(time.time()),
            "user_id": metadata.get("user_id")
        })

        # Append new metadata
        existing_data.append(metadata)

        # Write back to file
        with open(metadata_filepath, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
            
        print(f"Metadata saved successfully: {metadata}")
        return metadata_filepath
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")
        return None

#######################################################################################################
##################################### Initializing api calls ##########################################
##################################### get_available_models ############################################

# Endpoint to get and update available models
@app.route('/get_available_models', methods=['POST', 'GET'])
def get_available_models():
    global current_model_path, current_model_type

    try:
        if request.method == 'GET':
            # Fetch available models and current model
            current_model = next((model for model in AVAILABLE_MODELS
                                  if model["id"] == current_model_path), None)
            return jsonify({
                "models": AVAILABLE_MODELS,
                "current_model": current_model,
                "status": "success"
            })

        if request.method == 'POST':
            data = request.get_json()
            if not data or 'label' not in data:
                return jsonify({
                    "error": "No model label provided",
                    "status": "error"
                }), 400

            selected_model_label = data.get("label")
            selected_model = next((model for model in AVAILABLE_MODELS
                                   if model["label"] == selected_model_label), None)

            if not selected_model:
                return jsonify({
                    "error": f"Model with label '{selected_model_label}' not found",
                    "status": "error"
                }), 404

            # Update model path and type
            current_model_path = selected_model["id"]
            current_model_type = selected_model["type"]

            print(f"Loading model: {current_model_path} ({current_model_type})")
            success, message = load_model(current_model_path, current_model_type)

            if not success:
                return jsonify({
                    "error": f"Failed to load model: {message}",
                    "status": "error"
                }), 500

            return jsonify({
                "message": f"Model updated to '{selected_model_label}'",
                "current_model": selected_model,
                "status": "success"
            })

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
    
#######################################################################################################  
##################################### create_voice ####################################################
@app.route('/create_voice', methods=['POST'])
def create_voice():
    try:
        name = request.form.get('name')
        audio = request.files.get('audio')

        if not name or not audio:
            return jsonify({"status": "error", "message": "Name or audio file missing"}), 400

        filename_base = secure_filename(name)
        audio_path = os.path.join(SPEAKERS_DIR, filename_base + ".wav")
        text_path = os.path.join(SPEAKERS_DIR, filename_base + ".txt")

        # Save audio file
        audio.save(audio_path)

        # Save default reference phrase
        default_phrase = "Apples are healthy for you. The quick brown fox jumps over the lazy dog.."
        with open(text_path, "w") as f:
            f.write(default_phrase)

        # Add to available voices
        voice_entry = {
            "label": name,
            "id": audio_path,
            "text": text_path
        }
        AVAILABLE_VOICES.append(voice_entry)

        return jsonify({"status": "success", "message": f"Voice '{name}' created successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

#######################################################################################################
##################################### get_available_voices ############################################

@app.route('/get_available_voices', methods=['GET', 'POST'])
def get_available_voices():
    global current_voice_path
    try:
        if request.method == 'GET':
            current_voice = next((v for v in AVAILABLE_VOICES if v["id"] == current_voice_path), None)
            return jsonify({
                "voices": AVAILABLE_VOICES,
                "current_voice": current_voice,
                "status": "success"
            })

        elif request.method == 'POST':
            data = request.get_json()
            selected_label = data.get("label")
            selected = next((v for v in AVAILABLE_VOICES if v["label"] == selected_label), None)

            if not selected:
                return jsonify({"status": "error", "message": "Voice not found"}), 404

            current_voice_path = selected["id"]
            return jsonify({
                "status": "success",
                "message": f"Voice switched to {selected_label}",
                "current_voice": selected
            })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
##############################################################################################  
#################################### debug_voices ############################################
@app.route('/debug_voices', methods=['GET'])
def debug_voices():
    return jsonify({
        "available_voices": AVAILABLE_VOICES,
        #"current_voice": current_voice_path
    })

#######################################################################################################
##################################### / to test if backend is working #################################

@app.route('/')
def hello_world():
    return "<h1>Hey there! <br> Backend script running here!!</h1>"

#######################################################################################################
##################################### save_metadata_endpoint ##########################################

@app.route('/save_metadata', methods=['POST'])
def save_metadata_endpoint():
    """Endpoint to save arbitrary metadata to a JSON file."""
    metadata = request.get_json()

    if not metadata:
        return jsonify({"error": "No metadata provided"}), 400

    # Generate a random number and get the current epoch time
    random_number = random.randint(1000, 9999)  # Adjust the range as needed
    current_epoch_time = int(time.time())
    
    # Create the id as randomnumber+current epochtime
    metadata['id'] = f"{random_number}_{current_epoch_time}"
    
    # print(metadata)
    data =  save_metadata(metadata)
    # print(data)
    return jsonify({"status": "Success"}), 200

#######################################################################################################
##################################### remove_audio_s3 ##########################################

@app.route('/remove_audio_s3', methods=['POST'])
def delete_from_s3():
    """Helper function to delete multiple files from S3 using only file names."""
    s3_client = boto3.client('s3')
    # Construct the S3 keys from filenames
    inputJSON = request.get_json()
    objects_to_delete = [{'Key': f'audio/{filename}'} for filename in inputJSON['files']]
    try:
        response = s3_client.delete_objects(
            Bucket=S3_BUCKET,
            Delete={
                'Objects': objects_to_delete
            }
        )
        # Check if any files were successfully deleted
        deleted_files = response.get('Deleted', [])
        if deleted_files:
            print(f"Deleted files: {[obj['Key'] for obj in deleted_files]}")
            return jsonify({"status": "Success"}), 200
        else:
            print("No files were deleted.")
            return jsonify({"status": "failed"}), 200
        # Check for errors
        if 'Errors' in response:
            for error in response['Errors']:
                print(f"Error deleting {error['Key']}: {error['Message']}")
    except NoCredentialsError:
        print("Credentials not available")
    except ClientError as e:
        print(f"Error deleting from S3: {str(e)}")
        
#######################################################################################################
##################################### remove_record ###################################################

@app.route('/remove_record', methods=['POST'])
def remove_record_by_id(filename="metadata.json"):
    """
    Remove a specific record from the JSON file by matching the provided id.
    """
    inputJSON = request.get_json()

    # Define the path to the JSON file
    metadata_filepath = os.path.join(os.getcwd(), filename)

    # Check if the file exists
    if not os.path.exists(metadata_filepath):
        print(f"No file found at {metadata_filepath}")
        return jsonify({"status": "failed", "message": "Metadata file not found"}), 404

    # Load existing metadata
    try:
        with open(metadata_filepath, 'r') as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError:
        print("File is empty or corrupted.")
        return jsonify({"status": "failed", "message": "Metadata file corrupted"}), 400

    # Filter out the record with the matching id
    updated_data = [record for record in data if record.get("id") != inputJSON.get('id')]

    # Write the updated data back to the file
    try:
        with open(metadata_filepath, 'w') as json_file:
            json.dump(updated_data, json_file, indent=4)

        # Collect files to delete if specified
        filesList = [inputJSON.get('inputFile'), inputJSON.get('outputFile')]
        filesList = [file for file in filesList if file]  # Filter out None values

        if filesList:
            objects_to_delete = [{'Key': f'audio/{filename}'} for filename in filesList]
            response = s3_client.delete_objects(
                Bucket=S3_BUCKET,
                Delete={'Objects': objects_to_delete}
            )
            
            # Check if any files were successfully deleted
            deleted_files = response.get('Deleted', [])
            if deleted_files:
                print(f"Deleted files: {[obj['Key'] for obj in deleted_files]}")
            else:
                print("No files were deleted.")

        print(f"Record with id '{inputJSON['id']}' removed from {metadata_filepath}")
        return jsonify({"status": "Success", "message": f"Record with id '{inputJSON['id']}' removed"}), 200

    except Exception as e:
        print(f"Error updating metadata: {str(e)}")
        return jsonify({"status": "failed", "message": str(e)}), 500
        
#######################################################################################################
##################################### Saving to log ###################################################

# Function to create a log decorator
def log_to_file(route_func):
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Create a log file with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'logs/audio_process_{timestamp}.txt'
        
        # Capture stdout
        captured_output = StringIO()
        
        # Write header to log file
        with open(log_filename, 'w') as log_file:
            log_file.write(f"=== Audio Processing Log - {datetime.datetime.now()} ===\n\n")
        
        # Capture all stdout and redirect to our string buffer
        with redirect_stdout(captured_output):
            # Call the original function
            try:
                result = route_func(*args, **kwargs)
                success = True
            except Exception as e:
                print(f"Exception occurred: {str(e)}")
                print(traceback.format_exc())
                success = False
                result = {"error": str(e)}, 500
        
        # Get the captured output
        output = captured_output.getvalue()
        
        # Write to log file
        with open(log_filename, 'a') as log_file:
            log_file.write(output)
            log_file.write("\n\n=== Result ===\n")
            log_file.write(f"Success: {success}\n")
            if hasattr(result, 'json'):
                log_file.write(f"Response: {result.json}\n")
            else:
                log_file.write(f"Response: {result}\n")
            log_file.write(f"\n=== End of Log ===\n")
        
        print(f"Log saved to: {log_filename}")
        
        return result
    
    return wrapper

#######################################################################################################
##################################### process_audio ###################################################

# Apply the decorator to your route
@app.route('/process_audio', methods=['POST'])
@log_to_file # Apply the decorator to your route
def process_audio():
    # Declare global variables accessed and potentially modified within the function
    global current_model_path, model, processor, pipe, asr_pipeline

    print("\n=== Starting new audio processing request ===")
    response_start_time = time.time() # Start timing early to capture all operations

    # --- 1. Validate Audio File Upload ---
    if 'audio' not in request.files:
        print("Error: No audio file uploaded.")
        return jsonify({"error": "No audio file uploaded"}), 400

    try:
        # --- 2. Retrieve Request Parameters and Initialize Variables ---
        user_id = request.form.get('user_id', 'NO_ID')
        current_epoch_time = int(time.time())
        selected_model = request.form.get('model')
        phrase_text = None
        text_filename = None
        word_accuracy = None
        tmp_audio_path = None # Initialize to ensure it's always defined for cleanup

        print(f"Current model path before processing: {current_model_path}")
        print("Form data received:", dict(request.form))
        print(f"Selected model from form: {selected_model}")

        # --- 3. Model Loading and Validation ---
        if selected_model:
            # Find the exact model from AVAILABLE_MODELS
            matching_model = next((m for m in AVAILABLE_MODELS if m["id"] == selected_model), None)

            if matching_model:
                print(f"Found matching model: {matching_model['id']} of type: {matching_model['type']}")
                if selected_model != current_model_path:
                    success, message = False, "Unknown model type" # Default for safety

                    if matching_model["type"] == "whisper":
                        print(f"Loading new Whisper model: {selected_model}")
                        success, message = load_whisper_model(selected_model)
                    elif matching_model["type"] == "wav2vec2":
                        print(f"Loading new Wav2Vec2 model: {selected_model}")
                        success, message = load_wav2vec2_model(selected_model)
                    else:
                        print(f"Unknown model type requested: {matching_model['type']}")
                        return jsonify({"error": f"Invalid model type: {matching_model['type']}"}), 400

                    if success:
                        current_model_path = selected_model
                        print(f"Successfully switched to model: {current_model_path}")
                    else:
                        print(f"Failed to load model '{selected_model}': {message}")
                        return jsonify({"error": f"Failed to load model: {message}"}), 500
                else:
                    print("Selected model is already loaded, no need to reload.")
            else:
                print(f"Warning: Selected model '{selected_model}' not found in AVAILABLE_MODELS.")
                return jsonify({"error": f"Invalid model selection: {selected_model}"}), 400
        else:
            print("No 'model' provided in form data. Using currently loaded model.")
            # If no model selected, ensure current_model_path is set to a valid, loaded model.
            # This assumes `current_model_path` holds a valid ID and `model/processor/pipe/asr_pipeline` are initialized.
            # If `current_model_path` can be empty initially, you might need a default load here.
            pass # Use the existing current_model_path

        # Determine the effective model ID for logging/response
        effective_model_id = selected_model if selected_model else current_model_path
        effective_matching_model = next((m for m in AVAILABLE_MODELS if m["id"] == effective_model_id), None)
        model_label_for_response = effective_matching_model["label"] if effective_matching_model else effective_model_id

        print(f"Model used for processing: {current_model_path}")

        # --- 4. Handle Reference Text File (if provided) ---
        print("Files received:", list(request.files.keys()))
        if 'text_file' in request.files:
            text_file = request.files['text_file']
            if text_file.filename and text_file.filename.strip() != '':
                try:
                    text_filename = secure_filename(text_file.filename)
                    phrase_text = text_file.read().decode('utf-8').strip()
                    print(f"Reference text content: '{phrase_text}'")

                    # Save text file to S3
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_text_file:
                        text_file.seek(0) # Rewind the file pointer after reading
                        tmp_text_file.write(text_file.read())
                        tmp_text_file.flush()
                        text_s3_filename = f"{user_id}_text_{current_epoch_time}.txt"
                        text_s3_url = upload_to_s3(tmp_text_file.name, text_s3_filename, S3_BUCKET)
                        print(f"Text file uploaded to S3: {text_s3_url}")
                    os.remove(tmp_text_file.name) # Clean up temporary text file
                except Exception as e:
                    print(f"Error processing text file: {str(e)}")
                    traceback.print_exc() # Print full traceback for debugging
                    return jsonify({"error": f"Error processing text file: {str(e)}"}), 500

        # --- 5. Process Uploaded Audio File ---
        audio_file = request.files['audio']
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        print(f"Audio file extension: {file_extension}")

        supported_formats = {'.wav', '.mp3', '.m4a', '.mp4', '.weba'}

        if file_extension not in supported_formats and file_extension != '': # Allow empty string for inferred type
            print(f"Error: Unsupported file format '{file_extension}'. Supported: {', '.join(supported_formats)}")
            return jsonify({"error": f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"}), 400

        # Create a temporary file for the uploaded audio
        uploaded_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension if file_extension else '.tmp') as tmp_upload_file:
                audio_file.save(tmp_upload_file.name)
                uploaded_audio_path = tmp_upload_file.name
            print(f"Uploaded audio saved to temporary path: {uploaded_audio_path}")

            # Convert to WAV if not already WAV or if format was initially unknown
            if file_extension != '.wav':
                print(f"Converting '{file_extension}' to WAV format...")
                audio = None
                if file_extension == '.m4a':
                    audio = AudioSegment.from_file(uploaded_audio_path, format='m4a')
                elif file_extension == '.weba' or file_extension == '.mp4' or file_extension == '.tmp': # weba/mp4 are webm audio, or unkonwn
                    audio = AudioSegment.from_file(uploaded_audio_path, format='webm')
                elif file_extension == '.mp3':
                    audio = AudioSegment.from_file(uploaded_audio_path, format='mp3')
                else: # Fallback for other potential formats, or if file_extension was initially empty
                    audio = AudioSegment.from_file(uploaded_audio_path)

                tmp_audio_path = os.path.join(tempfile.gettempdir(), f"converted_{user_id}_{current_epoch_time}.wav")
                audio.export(tmp_audio_path, format='wav')
                print(f"Audio converted to WAV: {tmp_audio_path}")
                os.remove(uploaded_audio_path) # Clean up original uploaded file
            else:
                tmp_audio_path = uploaded_audio_path # Already a WAV file

        except Exception as e:
            if uploaded_audio_path and os.path.exists(uploaded_audio_path):
                os.remove(uploaded_audio_path) # Ensure cleanup if conversion fails
            print(f"Error during audio file handling/conversion: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Error handling audio file: {str(e)}"}), 500

        # --- 6. Preprocess Audio for ASR Model ---
        try:
            print("Preprocessing audio for ASR model...")

            audio_array, sampling_rate = librosa.load(tmp_audio_path, sr=16000)
            print(f"Original audio length: {len(audio_array)} samples at {sampling_rate}Hz")

            # Step 1: Noise Reduction
            #print("Applying noise reduction...")
            #audio_array = nr.reduce_noise(y=audio_array, sr=sampling_rate)

            # Step 2: Silence Removal
            #print("Removing silent gaps...")
            #audio_array, _ = librosa.effects.trim(audio_array, top_db=20)

            # Step 3: Padding if too short
            min_audio_length = 16000  # 1 sec
            if len(audio_array) < min_audio_length:
                print(f"Audio too short ({len(audio_array)} samples), padding to {min_audio_length} samples.")
                padding = np.zeros(min_audio_length - len(audio_array), dtype=audio_array.dtype)
                audio_array = np.concatenate([audio_array, padding])

            # Step 4: Frame padding
            frame_size = 600
            remainder = len(audio_array) % frame_size
            if remainder != 0:
                padding_needed = frame_size - remainder
                print(f"Padding audio to be a multiple of {frame_size} frames (adding {padding_needed} samples).")
                padding = np.zeros(padding_needed, dtype=audio_array.dtype)
                audio_array = np.concatenate([audio_array, padding])

            # Save processed audio
            sf.write(tmp_audio_path, audio_array, 16000)
            print(f"Preprocessed audio saved. Final length: {len(audio_array)} samples.")


        except Exception as e:
            print(f"Error during audio preprocessing: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Error preprocessing audio: {str(e)}"}), 500

        # Upload input audio file to S3
        input_filename_s3 = f"{user_id}_input_{current_epoch_time}.wav"
        input_audio_s3_url = upload_to_s3(tmp_audio_path, input_filename_s3, S3_BUCKET)
        if not input_audio_s3_url:
            raise Exception("Failed to upload input audio file to S3")
        print(f"Input audio uploaded to S3: {input_audio_s3_url}")

        # --- 7. Perform Speech-to-Text (ASR) Transcription ---
        print("Starting transcription...")
        transcription_start_time = time.time()
        transcription_text = ""

        try:
            if effective_matching_model["type"] == "whisper":
                # Ensure 'pipe' is correctly initialized for Whisper
                if pipe is None:
                    raise Exception("Whisper pipeline is not initialized.")
                result = pipe(tmp_audio_path, generate_kwargs={"language": "english"})
            elif effective_matching_model["type"] == "wav2vec2":
                # Ensure 'asr_pipeline' is correctly initialized for Wav2Vec2
                if asr_pipeline is None:
                    raise Exception("Wav2Vec2 pipeline is not initialized.")
                result = asr_pipeline(tmp_audio_path)
            else:
                raise Exception(f"Unsupported model type for transcription: {effective_matching_model['type']}")

            transcription_text = result['text'].strip()
            transcription_time = time.time() - transcription_start_time
            print(f"Transcription completed in {transcription_time:.2f} seconds.")
            print(f"Transcribed text: '{transcription_text}'")
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Error during transcription: {str(e)}"}), 500

        # --- 8. Post-process Transcription ---
        # Note: 'remove_repeated_phrases' was commented out in your original code.
        # If you intend to use it, uncomment. I'll keep it commented as per your original.
        # transcription_text = remove_repeated_phrases(transcription_text, n=1)

        transcription_text = correct_grammar(transcription_text)
        transcription_text = remove_filler_words(transcription_text)

        # Add a period at the end if missing for better TTS pronunciation
        if not transcription_text.endswith(('.', '!', '?')):
            transcription_text += '.'
        
        print(f"Post-processed Transcription Text: '{transcription_text}'")

        # --- 9. Calculate Word Accuracy (if reference text is available) ---
        if phrase_text:
            word_accuracy = calculate_word_accuracy(phrase_text, transcription_text)
            print(f"Word Accuracy: {word_accuracy:.2f}%")

        # --- 10. TTS-Processing using F5-TTS ---
        # print("Starting TTS model loading and audio generation...")
        # tts_start_time = time.time()
        # output_filename_s3 = f"{user_id}_output_{current_epoch_time}.wav"
        # output_path = os.path.join(tempfile.gettempdir(), output_filename_s3)

        # try:
        #     if not transcription_text or transcription_text.strip() == "":
        #         raise Exception("transcription_text is missing or empty.")

        #     # Always use AVAILABLE_VOICES[0] as reference
        #     if not AVAILABLE_VOICES:
        #         raise Exception("No reference voices available.")

        #     speaker_entry = AVAILABLE_VOICES[0]
        #     ref_audio_path = speaker_entry["id"]

        #     with open(speaker_entry["text"], "r") as f:
        #         ref_text = f.read().strip()

        #     print(f"Using hardcoded speaker: {speaker_entry['label']}")

        #     if os.path.exists(output_path):
        #         os.remove(output_path)

        #     cmd = [
        #         "f5-tts_infer-cli",
        #         "--model", "F5TTS_v1_Base",
        #         "--ref_audio", ref_audio_path,
        #         "--ref_text", ref_text,
        #         "--gen_text", transcription_text,
        #         "--output_file", output_path
        #     ]

        #     print(f"Running F5-TTS CLI: {' '.join(cmd)}")
        #     result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        #     print("F5-TTS CLI output:\n", result.stdout)

        #     if not os.path.exists(output_path):
        #         raise Exception("Expected output audio not found at specified location.")

        #     print(f"Generated TTS audio saved to: {output_path}")
        #     output_audio_s3_url = upload_to_s3(output_path, output_filename_s3, S3_BUCKET)
        #     if not output_audio_s3_url:
        #         raise Exception("Failed to upload output audio file to S3.")
        #     print(f"Output audio uploaded to S3: {output_audio_s3_url}")

        #     tts_time = time.time() - tts_start_time
        #     print(f"TTS processing completed in {tts_time:.2f} seconds.")

        # except subprocess.CalledProcessError as e:
        #     print(f"F5-TTS CLI error: {e.stderr or e.stdout or str(e)}")
        #     import traceback; traceback.print_exc()
        #     print("Creating fallback silent audio file...")
        #     fallback_audio = np.zeros(24000, dtype=np.float32)
        #     sf.write(output_path, fallback_audio, samplerate=24000)
        #     output_audio_s3_url = upload_to_s3(output_path, output_filename_s3, S3_BUCKET)
        #     print(f"Fallback silent audio uploaded: {output_audio_s3_url}")

        # except Exception as e:
        #     print(f"Error in TTS processing: {str(e)}")
        #     import traceback; traceback.print_exc()
        #     print("Creating fallback silent audio file...")
        #     fallback_audio = np.zeros(24000, dtype=np.float32)
        #     sf.write(output_path, fallback_audio, samplerate=24000)
        #     output_audio_s3_url = upload_to_s3(output_path, output_filename_s3, S3_BUCKET)
        #     print(f"Fallback silent audio uploaded: {output_audio_s3_url}")

        # finally:
        #     for path in [tmp_audio_path, output_path, uploaded_audio_path]:
        #         if path and os.path.exists(path):
        #             os.remove(path)

        # --- 10. TTS-Processing using F5-TTS ---
        print("Starting TTS model loading and audio generation...")
        tts_start_time = time.time()
        output_filename_s3 = f"{user_id}_output_{current_epoch_time}.wav"
        output_path = os.path.join(tempfile.gettempdir(), output_filename_s3)

        try:
            # Ensure valid transcription
            if not transcription_text or transcription_text.strip() == "":
                raise Exception("transcription_text is missing or empty.")

            # Determine reference audio and text based on speaker selection
            if current_voice_path:
                # Find speaker entry
                speaker_entry = next((v for v in AVAILABLE_VOICES if v["id"] == current_voice_path), None)
                if not speaker_entry:
                    raise Exception("Selected speaker voice not found in available voices.")

                ref_audio_path = speaker_entry["id"]
                with open(speaker_entry["text"], "r") as f:
                    ref_text = f.read().strip()

                print(f"Using speaker voice: {speaker_entry['label']}")
            else:
                if not tmp_audio_path or not os.path.exists(tmp_audio_path):
                    raise Exception("Reference audio is missing or invalid.")
                ref_audio_path = tmp_audio_path
                ref_text = phrase_text.strip() if phrase_text and phrase_text.strip() else transcription_text

                print("Using temporary reference audio and dynamic phrase.")

            # Clean old output
            if os.path.exists(output_path):
                os.remove(output_path)

            # Build TTS command
            cmd = [
                "f5-tts_infer-cli",
                "--model", "F5TTS_v1_Base",
                "--ref_audio", ref_audio_path,
                "--ref_text", ref_text,
                "--gen_text", transcription_text,
                "--output_file", output_path
            ]

            print(f"Running F5-TTS CLI: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("F5-TTS CLI output:\n", result.stdout)

            if not os.path.exists(output_path):
                raise Exception("Expected output audio not found at specified location.")

            print(f"Generated TTS audio saved to: {output_path}")
            output_audio_s3_url = upload_to_s3(output_path, output_filename_s3, S3_BUCKET)
            if not output_audio_s3_url:
                raise Exception("Failed to upload output audio file to S3.")
            print(f"Output audio uploaded to S3: {output_audio_s3_url}")

            tts_time = time.time() - tts_start_time
            print(f"TTS processing completed in {tts_time:.2f} seconds.")

        except subprocess.CalledProcessError as e:
            print(f"F5-TTS CLI error: {e.stderr or e.stdout or str(e)}")
            import traceback; traceback.print_exc()
            print("Creating fallback silent audio file...")
            fallback_audio = np.zeros(24000, dtype=np.float32)
            sf.write(output_path, fallback_audio, samplerate=24000)
            output_audio_s3_url = upload_to_s3(output_path, output_filename_s3, S3_BUCKET)
            print(f"Fallback silent audio uploaded: {output_audio_s3_url}")

        except Exception as e:
            print(f"Error in TTS processing: {str(e)}")
            import traceback; traceback.print_exc()
            print("Creating fallback silent audio file...")
            fallback_audio = np.zeros(24000, dtype=np.float32)
            sf.write(output_path, fallback_audio, samplerate=24000)
            output_audio_s3_url = upload_to_s3(output_path, output_filename_s3, S3_BUCKET)
            print(f"Fallback silent audio uploaded: {output_audio_s3_url}")

        finally:
            for path in [tmp_audio_path, output_path, uploaded_audio_path]:
                if path and os.path.exists(path):
                    os.remove(path)



        # --- 11. Prepare and Save Metadata ---
        total_response_time = time.time() - response_start_time
        metadata = {
            "id": f"{random.randint(1000, 9999)}_{current_epoch_time}",
            "user_id": user_id,
            "inputFile": input_filename_s3, # Use the S3 filename
            "outputFile": output_filename_s3, # Use the S3 filename
            "Transcription": transcription_text,
            "model_used": model_label_for_response, # Use the human-readable label
            "duration": current_epoch_time, # This usually refers to audio duration, not epoch time. Recheck logic.
                                            # If it's epoch time, maybe rename to 'request_timestamp'.
            "processing_time": {
                "transcription_s": round(transcription_time, 2),
                "tts_s": round(tts_time, 2) if 'tts_time' in locals() else None,
                "total_s": round(total_response_time, 2)
            }
        }

        if phrase_text is not None:
            metadata.update({
                "phrase_text": phrase_text,
                "input_file": text_filename, # Clarified key name
                "word_accuracy": round(word_accuracy, 2) if word_accuracy is not None else None
            })

        save_metadata(metadata)
        print("Metadata saved successfully.")

        # --- 12. Prepare and Send Response ---
        response_data = metadata.copy()
        response_data["message"] = "Process completed successfully"

        print("Final response data:", response_data)
        print("=== Audio processing completed ===\n")
        return jsonify(response_data)

    except Exception as e:
        # Catch any unexpected errors that were not handled earlier
        print(f"An unhandled error occurred during audio processing: {str(e)}")
        traceback.print_exc()
        # Ensure temporary files are cleaned up even on unhandled exceptions
        if 'tmp_audio_path' in locals() and tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        if 'output_path' in locals() and output_path and os.path.exists(output_path):
            os.remove(output_path)
        if 'uploaded_audio_path' in locals() and uploaded_audio_path and os.path.exists(uploaded_audio_path):
            os.remove(uploaded_audio_path)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

######################################################################################################
##################################### debug_models ###################################################

# Add this debugging route to verify available models
@app.route('/debug_models', methods=['GET'])
def debug_models():
    return jsonify({
        "available_models": AVAILABLE_MODELS,
        "current_model": current_model_path
    })

#######################################################################################################
##################################### update_filename #################################################

@app.route('/update_filename', methods=['POST'])
def update_filename():
    """Endpoint to rename an existing file in S3."""
    data = request.json
    old_filename = data.get('old_filename')
    new_filename = data.get('new_filename')

    if not old_filename or not new_filename:
        return jsonify({"error": "Both old and new file names are required."}), 400

    # Rename the file in S3
    new_s3_url = rename_s3_file(S3_BUCKET, old_filename, new_filename)
    if new_s3_url:
        return jsonify({
            "message": "File renamed successfully",
            "new_file_url": new_s3_url
        })
    else:
        return jsonify({"error": "File renaming failed."}), 500

#######################################################################################################
##################################### store_user ######################################################

@app.route('/store_user', methods=['POST'])
def handle_store_user():
    """
    Endpoint to store user details from Google login.

    Expects a JSON payload with user information including 'sub', 'email', 
    'email_verified', 'name', 'given_name', 'family_name', and 'picture'.
    """
    # Call the utility function to store user data
    response = store_user()  
    # Return the response from the utility function
    return response  

#####################################################################################################
##################################### temp_url ######################################################

@app.route('/temp_url', methods=['GET'])
def get_file_url():
    file_name = request.args.get('fileName')

    if not file_name:
        return jsonify({"error": "Missing 'fileName' parameter"}), 400

    try:
        # Check if the file exists in S3 with proper key
        s3_client.head_object(Bucket=S3_BUCKET, Key=f"audio/{file_name}")
        
        # Generate presigned URL using the correct S3 client
        signed_url = create_presigned_url(S3_BUCKET, f"audio/{file_name}")

        if signed_url:
            return jsonify({"temp_URL": signed_url})
        else:
            return jsonify({"error": f"Failed to generate signed URL for '{file_name}'"}), 500

    except NoCredentialsError:
        return jsonify({"error": "AWS credentials are not available."}), 500
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return jsonify({"error": f"File '{file_name}' not found in S3 bucket."}), 404
        else:
            return jsonify({"error": "Error checking file in S3."}), 500

#######################################################################################################
##################################### get_metadata ####################################################

@app.route('/get_metadata/<user_id>', methods=['GET'])
def get_user(user_id):
    """Retrieve user details based on user_id."""
    stored_data = load_user_data()

    # Check if the user_id exists
    if user_id in stored_data:
        return jsonify({
            "user_id": user_id,
            "details": stored_data[user_id]
        }), 200
    else:
        return jsonify({"error": "User not found"}), 404

#######################################################################################################
##################################### get_user_records ################################################

@app.route('/get_user_records/<user_id>', methods=['GET'])
def get_user_records(user_id):
    """Endpoint to retrieve metadata records for a specific user based on user_id."""
    metadata_filename = "metadata.json"
    metadata_filepath = os.path.join(os.getcwd(), metadata_filename)

    if not os.path.exists(metadata_filepath):
        return jsonify({"user_records": []}), 200

    try:
        with open(metadata_filepath, 'r') as json_file:
            records_metadata = json.load(json_file)

        # Filter records for the given user_id and ensure all fields are present
        user_records = []
        print("[RECORDS METADATA]", records_metadata)
        for record in records_metadata:

            if record.get("user_id") == user_id:
                # Make sure all required fields are present
                print("[RECORD]")
                print(record)
                processed_record = {
                    "id": record.get("id"),
                    "user_id": record.get("user_id"),
                    "inputFile": record.get("inputFile"),
                    "outputFile": record.get("outputFile"),
                    "Transcription": record.get("Transcription"),
                    "phraseText": record.get("phrase_text"),
                    "fileName": record.get("input_file"),
                    "accuracy": record.get("word_accuracy"),
                    "model_used": record.get("model_used"),
                    "timestamp": record.get("duration"),
                    "processing_time": record.get("processing_time", {})
                }
                user_records.append(processed_record)

        if user_records:
            return jsonify({"user_records": user_records}), 200
        else:
            return jsonify({"user_records": [], "message": "No records found for the specified user"}), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding metadata file."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

#######################################################################################################
##################################### get_records_metadata ############################################

@app.route('/get_records_metadata', methods=['GET'])
def get_records_metadata():
    """Endpoint to retrieve metadata of all uploaded records."""
    metadata_filename = "metadata.json"
    metadata_filepath = os.path.join(os.getcwd(), metadata_filename)

    # Check if the metadata file exists
    if not os.path.exists(metadata_filepath):
        return jsonify({"records_metadata": []}), 200  # Return empty list if no metadata found

    try:
        # Load the metadata from the file
        with open(metadata_filepath, 'r') as json_file:
            records_metadata = json.load(json_file)
            return jsonify({"records_metadata": records_metadata}), 200
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding metadata file."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#######################################################################################################
##################################### main ############################################

if __name__ == '__main__':
    app.run(debug=False, host='192.168.0.49', port= 5050)