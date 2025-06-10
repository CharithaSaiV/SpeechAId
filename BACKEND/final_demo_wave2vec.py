import os
import random
import json
import tempfile
import whisper
import time
import psutil
import boto3
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from TTS.api import TTS
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from utils.user_storage import store_user, load_user_data
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
import librosa
import torch
import os
from tqdm import tqdm
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import io
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize



#=============================================================================================
app = Flask(__name__, static_folder="./build", static_url_path="/")
# CORS(app) "http://192.168.0.49:5020", "http://192.168.1.81:5020", "http://localhost:5020
# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://localhost:3001/","http://localhost:3000/","http://38.188.108.234:5020","http://192.168.0.49:5020","https://speechaid.convogene.ai:5020/","http://38.188.108.234","http://192.168.0.49:5020","https://speechaid.convogene.ai:5020/api/store_user"],
#         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"],
#         "supports_credentials": True
#      }
#     })
CORS(app, resources={r"/*": {"origins": "*"}}) 

# # Allow the frontend to make API requests from the public IP (38.188.108.234:5020)
# CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})
# # CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})
socketio = SocketIO(app, cors_allowed_origins="*")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#=============================================================================================
# Import necessary modules
import os
import time
import torch
from flask import Flask, jsonify, request
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)
from datasets import load_dataset
from dotenv import load_dotenv
import boto3
import language_tool_python
from nltk.util import ngrams
from TTS.api import TTS

import re
from pydub import AudioSegment
from kokoro import KPipeline

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

# Load environment variables
load_dotenv()

# AWS S3 configuration
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

# Global settings
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Define available models with `type` differentiation
AVAILABLE_MODELS = [
    {"id": "openai/whisper-small", "label": "Whisper Small Base", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/finetuned_phrasecards", "label": "Whisper Fine-tuned on Phrasecards", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/finetuned_m01", "label": "Whisper Fine-tuned on M01", "type": "whisper"},
    #{"id": "facebook/wav2vec2-large-960h", "label": "Wav2Vec2 Large", "type": "wav2vec2"},
    #{"id": "/home/arun/ranga-ai/active-speech/testrun/Model/wav2vec-phrasecards", "label": "Wav2Vec2 Fine-tuned on Phrasecards", "type": "wav2vec2"},
    #{"id": "/home/arun/ranga-ai/active-speech/testrun/Model/wav2vec-uaspeech", "label": "Wav2Vec2 Fine-tuned on UASpeech", "type": "wav2vec2"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/Synthetic_Phrasecards", "label": "Whisper Fine-tuned on Phrasecards and Synthetic data", "type": "whisper"},
    #{"id": "/home/arun/ranga-ai/active-speech/testrun/Model/Synthetic_Finetuned_V1", "label": "Whisper Fine-tuned on Synthetic data V1", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/Synthetic_Finetuned_V2", "label": "Whisper Fine-tuned on Synthetic data V2", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/GR_Model/", "label": "Whisper Fine-tuned on GR_V1", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/GR_Model_new/", "label": "Whisper Fine-tuned on GR_V2", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/GR_V3/", "label": "Whisper Fine-tuned on GR_V3", "type": "whisper"},
    #{"id": "/home/arun/ranga-ai/active-speech/testrun/Model/GR_V4_Cha1/", "label": "Whisper Fine-tuned on GR_V4_Cha1", "type": "whisper"},
    {"id": "/home/arun/ranga-ai/active-speech/testrun/Model/GR_V4/", "label": "Whisper Fine-tuned on GR_V4", "type": "whisper"}
    
    
    
]

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

# Load TTS model
#synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=0)
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
'''def load_tts_model():
    print("Loading Coqui TTS model...")
    try:
        # Initialize TTS with a pre-trained English model
        tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        
        return tts_model
    except Exception as e:
        print(f"Error loading Kokoro TTS model: {str(e)}")
        traceback.print_exc()
        return None
    
synthesiser = load_tts_model()'''
def load_tts_model():
    """
    Load the Kokoro TTS model.
    
    Returns:
        KPipeline: Initialized Kokoro TTS pipeline
    """
    print("Loading Kokoro TTS model...")
    try:
        # SInitialize Kokoro TTS pipeline
        # You can customize the language code and other parameters as needed
        pipeline = KPipeline(lang_code='a')
        return pipeline
    except Exception as e:
        print(f"Error loading Kokoro TTS model: {str(e)}")
        traceback.print_exc()
        return None
synthesiser = KPipeline(lang_code='a')

# from transformers import AutoProcessor, AutoModel

# # Load Suno/Bark TTS model
# print("Loading Suno/Bark TTS model...")
# processor = AutoProcessor.from_pretrained("suno/bark")
# model = AutoModel.from_pretrained("suno/bark").to("cuda" if torch.cuda.is_available() else "cpu")
# print("Bark TTS model loaded.")

# Utility functions for text processing
tool = language_tool_python.LanguageTool('en-US')
##NLTK

def remove_repeated_phrases(text, n=1):
    tokens = text.split()
    seen = set()
    cleaned_tokens = []
    for i in range(len(tokens)):
        gram = tuple(tokens[i:i + n])  # Generate n-grams
        if gram not in seen:
            cleaned_tokens.append(tokens[i])
            seen.add(gram)
    return ' '.join(cleaned_tokens)

def correct_grammar(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

def remove_filler_words(text, fillers=["uh", "um"]):
    tokens = text.split()
    return ' '.join([word for word in tokens if word.lower() not in fillers])

def convert_to_wav(input_file_path):
    audio = AudioSegment.from_file(input_file_path)
    wav_path = input_file_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path
#=============================================================================================


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



#============================================================================================
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
    
def calculate_word_accuracy(reference, prediction):
    """Calculate word accuracy between reference and prediction texts."""
    wer_metric = load("wer")
    normalizer = BasicTextNormalizer()
    normalized_reference = normalizer(reference)
    normalized_prediction = normalizer(prediction)
    wer = wer_metric.compute(references=[normalized_reference], predictions=[normalized_prediction])
    word_accuracy = (1 - wer) * 100
    return word_accuracy
#=============================================================================================

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

#============================================================================================
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
#============================================================================================
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
#===========================================================================================


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
#=============================================================================================

@app.route('/')
def hello_world():
    return "<h1>Hey there! <br> Backend script running here!!</h1>"
#=======================================================================


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
#=============================================================================================
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
#=============================================================================================
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
#=============================================================================================
import sys
import os
import time
import datetime
from functools import wraps
from io import StringIO
from contextlib import redirect_stdout

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
                import traceback
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

# Apply the decorator to your route
@app.route('/process_audio', methods=['POST'])
@log_to_file
def process_audio():
    global current_model_path, model, processor, pipe, asr_pipeline
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    print("\n=== Starting new audio processing request ===")
    try:
        # Get basic information
        user_id = request.form.get('user_id', 'NO_ID')
        current_epoch_time = int(time.time())

        # Debug current state
        print(f"Current model path before processing: {current_model_path}")
        print("Form data received:", dict(request.form))

        # Get and validate the selected model
        selected_model = request.form.get('model')
        print(f"Selected model from form: {selected_model}")

        if selected_model:
            # Find the exact model from AVAILABLE_MODELS
            matching_model = next((model for model in AVAILABLE_MODELS if model["id"] == selected_model), None)
            if matching_model:
                print(f"Found matching model: {matching_model['id']}")
                if matching_model["type"] == "whisper":  # Check if the model type is "whisper"
                    if selected_model != current_model_path:
                        print(f"Loading new Whisper model: {selected_model} (current: {current_model_path})")
                        success, message = load_whisper_model(selected_model)  # Load Whisper model
                        if success:
                            current_model_path = selected_model
                            print(f"Successfully switched to model: {current_model_path}")
                        else:
                            print(f"Failed to load Whisper model: {message}")
                            return jsonify({"error": f"Failed to load model: {message}"}), 500
                    else:
                        print("Selected Whisper model is same as current model, no need to reload")

                elif matching_model["type"] == "wav2vec2":  # Check if the model type is "wav2vec2"
                    if selected_model != current_model_path:
                        print(f"Loading new Wav2Vec2 model: {selected_model} (current: {current_model_path})")
                        success, message = load_wav2vec2_model(selected_model)  # Load Wav2Vec2 model
                        if success:
                            current_model_path = selected_model
                            print(f"Successfully switched to model: {current_model_path}")
                        else:
                            print(f"Failed to load Wav2Vec2 model: {message}")
                            return jsonify({"error": f"Failed to load model: {message}"}), 500
                    else:
                        print("Selected Wav2Vec2 model is same as current model, no need to reload")
                else:
                    print(f"Unknown model type: {matching_model['type']}")
                    return jsonify({"error": f"Invalid model type: {matching_model['type']}"}), 400
            else:
                print(f"Warning: Selected model '{selected_model}' not found in AVAILABLE_MODELS")
                return jsonify({"error": f"Invalid model selection: {selected_model}"}), 400
        else:
            print("No model_id provided in form data")
            selected_model = current_model_path
            print(f"Using current model: {selected_model}")

        print(f"Final model being used for processing: {current_model_path}")

        # Handle text file and reference text
        phrase_text = None
        text_filename = None
        word_accuracy = None

        print("Files received:", list(request.files.keys()))

        if 'text_file' in request.files:
            text_file = request.files['text_file']
            print(f"Text file received: {text_file.filename}")
            if text_file.filename != '':
                try:
                    text_filename = secure_filename(text_file.filename)
                    # Read and decode text content
                    phrase_text = text_file.read().decode('utf-8').strip()
                    print(f"Reference text content: '{phrase_text}'")

                    # Save text file to S3
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_text_file:
                        text_file.seek(0)
                        tmp_text_file.write(text_file.read())
                        tmp_text_file.flush()
                        text_s3_filename = f"{user_id}_text_{current_epoch_time}.txt"
                        text_s3_url = upload_to_s3(tmp_text_file.name, text_s3_filename, S3_BUCKET)
                        print(f"Text file uploaded to S3: {text_s3_url}")
                    os.remove(tmp_text_file.name)
                except Exception as e:
                    print(f"Error processing text file: {str(e)}")
                    return jsonify({"error": f"Error processing text file: {str(e)}"}), 500


        audio_file = request.files['audio']
        response_start_time = time.time()
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        print (f'file extention is {file_extension}')
        print(f'audio file is {audio_file}')
        print(f'audio file name is {audio_file.name}')
        if file_extension == '':
           new_file=os.path.join(tempfile.gettempdir(), f'converted_{current_epoch_time}.weba')
           audio_file.save(new_file)
           audio = AudioSegment.from_file(new_file, format='webm')
           tmp_audio_path = os.path.join(tempfile.gettempdir(), f"converted_{current_epoch_time}.wav")
           audio.export(tmp_audio_path, format='wav')

        else:
           supported_formats = {'.wav', '.mp3', '.m4a', '.mp4', '.weba'}

           if file_extension not in supported_formats:
              return jsonify({"error": f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"}), 400

        # Create a temporary file for the uploaded audio
           with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_upload_file:
               audio_file.save(tmp_upload_file.name)
               uploaded_audio_path = tmp_upload_file.name

         #Convert to WAV if not already WAV
           if file_extension != '.wav':
               try:
                 print(f"Converting {file_extension} to WAV format...")
                 if file_extension == '.m4a':
        #            # Special handling for m4a files
                     audio = AudioSegment.from_file(uploaded_audio_path, format='m4a')
                 elif file_extension == '.weba':
        #            # Special handling for weba files
                        audio = AudioSegment.from_file(uploaded_audio_path, format='webm')  # weba is webm audio
                 else:
                    audio = AudioSegment.from_file(uploaded_audio_path)
        #           print('hello1')
        #           #Create temporary WAV file
                    tmp_audio_path = os.path.join(tempfile.gettempdir(), f"converted_{current_epoch_time}.wav")
                    audio.export(tmp_audio_path, format='wav')
        #           print('hello2')
        #           #Remove the original uploaded file
                    os.remove(uploaded_audio_path)
               except Exception as e:
                    os.remove(uploaded_audio_path)
                    print(f"Error converting audio: {str(e)}")
                    return jsonify({"error": f"Error converting audio format: {str(e)}"}), 500
           else:
              tmp_audio_path = uploaded_audio_path

        # # Process audio file
        # audio_file = request.files['audio']
        # response_start_time = time.time()

        # # Get file extension and validate format
        # file_extension = os.path.splitext(audio_file.filename)[1].lower()
        # supported_formats = {'.wav', '.mp3', '.m4a', '.mp4', '.weba'}

        # if file_extension not in supported_formats:
        #     return jsonify({"error": f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"}), 400

        # # Create a temporary file for the uploaded audio
        # with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_upload_file:
        #     audio_file.save(tmp_upload_file.name)
        #     uploaded_audio_path = tmp_upload_file.name

        # # Convert to WAV if not already WAV
        # if file_extension != '.wav':
        #     try:
        #         print(f"Converting {file_extension} to WAV format...")
        #         if file_extension == '.m4a':
        #             # Special handling for m4a files
        #             audio = AudioSegment.from_file(uploaded_audio_path, format='m4a')
        #         elif file_extension == '.weba':
        #             # Special handling for weba files
        #             audio = AudioSegment.from_file(uploaded_audio_path, format='webm')  # weba is webm audio
        #         else:
        #             audio = AudioSegment.from_file(uploaded_audio_path)
                
        #         # Create temporary WAV file
        #         tmp_audio_path = os.path.join(tempfile.gettempdir(), f"converted_{current_epoch_time}.wav")
        #         audio.export(tmp_audio_path, format='wav')
                
        #         # Remove the original uploaded file
        #         os.remove(uploaded_audio_path)
        #     except Exception as e:
        #         os.remove(uploaded_audio_path)
        #         print(f"Error converting audio: {str(e)}")
        #         return jsonify({"error": f"Error converting audio format: {str(e)}"}), 500
        # else:
        #     tmp_audio_path = uploaded_audio_path

        # NEW CODE: Preprocess audio to avoid tensor size mismatches
        try:
            print("Preprocessing audio file...")
            import numpy as np
            import librosa
            import soundfile as sf
            import traceback
            
            # Load and resample audio
            audio_array, sampling_rate = librosa.load(tmp_audio_path, sr=16000)
            print(f"Original audio length: {len(audio_array)} samples at {sampling_rate}Hz")
            
            # Check if audio is too short and pad if necessary
            min_audio_length = 16000  # 1 second at 16kHz
            if len(audio_array) < min_audio_length:
                print(f"Audio too short ({len(audio_array)} samples), padding to {min_audio_length} samples")
                padding = np.zeros(min_audio_length - len(audio_array))
                audio_array = np.concatenate([audio_array, padding])
            
            # Ensure audio length is a multiple of model's expected frame size
            # 600 and 681 from your error suggest a frame or window size issue
            frame_size = 600  # This is based on your error message
            remainder = len(audio_array) % frame_size
            if remainder != 0:
                padding_needed = frame_size - remainder
                print(f"Padding audio to be multiple of {frame_size} frames (adding {padding_needed} samples)")
                padding = np.zeros(padding_needed)
                audio_array = np.concatenate([audio_array, padding])
            
            # Save the preprocessed audio
            sf.write(tmp_audio_path, audio_array, 16000)
            print(f"Preprocessed audio length: {len(audio_array)} samples")
            
        except Exception as e:
            print(f"Error preprocessing audio: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Error preprocessing audio: {str(e)}"}), 500

        # Save the uploaded audio file temporarily
        input_filename = f"{user_id}_input_{current_epoch_time}.wav"

        # Upload input audio file to S3
        input_audio_s3_url = upload_to_s3(tmp_audio_path, input_filename, S3_BUCKET)
        if not input_audio_s3_url:
            raise Exception("Failed to upload input audio file to S3")
        
        print(f"Input audio uploaded: {input_audio_s3_url}")

        # Transcription using selected model (Whisper or Wav2Vec2)
        print("Starting transcription...")
        transcription_start_time = time.time()

        try:
            # Added try-except block around transcription for better error handling
            if matching_model["type"] == "whisper":
                result = pipe(tmp_audio_path, generate_kwargs={"language": "english"})
            elif matching_model["type"] == "wav2vec2":
                result = asr_pipeline(tmp_audio_path)
            transcription_text = result['text'].strip()
            
            transcription_time = time.time() - transcription_start_time
            print(f"Transcription completed in {transcription_time:.2f} seconds")
            print(f"Transcribed text: '{transcription_text}'")
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            print(f"Audio file: {audio_file.filename}, preprocessed length: {len(audio_array)}")
            traceback.print_exc()
            return jsonify({"error": f"Error during transcription: {str(e)}"}), 500

        # Post-processing (e.g., removing repeated phrases, correcting grammar)
        #transcription_text = remove_repeated_phrases(transcription_text, n=1)

        transcription_text = correct_grammar(transcription_text)
        transcription_text = remove_filler_words(transcription_text)
        if not transcription_text.endswith('.'):
            transcription_text += '.'

        print(f"Processed Transcription Text: '{transcription_text}'")

        # Calculate word accuracy if reference text is available
        if phrase_text:
            word_accuracy = calculate_word_accuracy(phrase_text, transcription_text)
            print(f"Word Accuracy: {word_accuracy:.2f}%")

        # Generate speech using Coqui TTS
        '''print("Generating speech using Coqui TTS...")
        tts_start_time = time.time()
        try:
            # Check if synthesiser was loaded successfully
            if synthesiser is None:
                raise Exception("TTS model was not loaded properly")
            
            # Save output audio directly to a temporary file
            output_filename = f"{user_id}_output_{current_epoch_time}.wav"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
            print(f"Processing transcription of length: {len(transcription_text)}")
            print(f"this is transcription going to tts:{(transcription_text)}")
            
            # Generate speech directly to file
            synthesiser.tts_to_file(text=transcription_text, file_path=output_path)
            
            print(f"Audio generated and saved to: {output_path}")
            
            # Verify the audio file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Generated audio file size: {os.path.getsize(output_path)} bytes")
            else:
                raise Exception("Generated audio file is empty or doesn't exist")
            
            # Upload to S3
            output_audio_s3_url = upload_to_s3(output_path, output_filename, S3_BUCKET)
            if not output_audio_s3_url:
                raise Exception("Failed to upload output audio file to S3")
            
            print(f"Output audio uploaded: {output_audio_s3_url}")
            tts_end_time = time.time()
            print(f"TTS processing completed in {tts_end_time - tts_start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error in TTS processing: {str(e)}")
            traceback.print_exc()
            # Create a fallback audio file with silence if TTS fails
            # This ensures the process continues even if TTS fails
            print("Creating fallback silent audio file...")
            fallback_audio = np.zeros(16000)  # 1 second of silence at 16kHz
            output_filename = f"{user_id}_output_{current_epoch_time}.wav"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            sf.write(output_path, fallback_audio, samplerate=16000)
            output_audio_s3_url = upload_to_s3(output_path, output_filename, S3_BUCKET)
            print(f"Fallback silent audio uploaded: {output_audio_s3_url}")

        # Clean up temporary files
        os.remove(tmp_audio_path)
        os.remove(output_path)'''
        
        #def split_text_into_chunks(text, max_chunk_length=100):
            #"""
            #Split long text into chunks of manageable length.
            
            #Args:
            #   text (str): Input text to be split
            #  max_chunk_length (int): Maximum number of words per chunk
            
            #Returns:
             #   list: List of text chunks
            
            # Split text into words
            #words = text.split()
            
            # Initialize chunks
            #chunks = []
            #current_chunk = []
            
            #for word in words:
            #   current_chunk.append(word)
                
                # If chunk reaches max length, add to chunks and reset
            #    if len(current_chunk) >= max_chunk_length:
            #       chunks.append(' '.join(current_chunk))
            #        current_chunk = []
            
            # Add any remaining words
            #if current_chunk:
            #    chunks.append(' '.join(current_chunk))
            
            #return chunks
        
        # Load Kokoro TTS model 
        #print("Loading Kokoro TTS model...") 
        #try: 
            # Initialize Kokoro TTS pipeline
        #   synthesiser = KPipeline(lang_code='a')
        #except Exception as e: 
        #   print(f"Error loading Kokoro TTS model: {str(e)}") 
        #    traceback.print_exc() 
        #    synthesiser = None
        
        # Generate speech using Kokoro TTS 
        #print("Generating speech using Kokoro TTS...") 
         
        
         

         
        try: 
            # Check if synthesiser was loaded successfully 
            tts_start_time = time.time()
            #if synthesiser is None: 
                #raise Exception("TTS model was not loaded properly") 
             
            # Save output audio directly to a temporary file 
            output_filename = f"{user_id}_output_{current_epoch_time}.wav" 
            output_path = os.path.join(tempfile.gettempdir(), output_filename) 
             
            print(f"Processing transcription of length: {len(transcription_text)}") 
            print(f"this is transcription going to tts:{(transcription_text)}") 
        
        
        #Generate speech using Microsoft TTS
        # print("Generating speech using TTS...")
        # tts_start_time = time.time()
        # speech = synthesiser(transcription_text, forward_params={"speaker_embeddings": speaker_embedding})
        # output_filename = f"{user_id}_output_{current_epoch_time}.wav"
        # output_path = os.path.join(tempfile.gettempdir(), output_filename)
        # sf.write(output_path, speech["audio"], samplerate=speech["sampling_rate"])
        # output_audio_s3_url = upload_to_s3(output_path, output_filename, S3_BUCKET)
        # if not output_audio_s3_url:
        #     raise Exception("Failed to upload output audio file to S3")
        # print(f"Output audio uploaded: {output_audio_s3_url}")
               #     # Split text into chunks
        #     #text_chunks = split_text_into_chunks(transcription_text)
            
        #     # List to store audio chunks
        #     #audio_chunks = []
            
        #     # Generate audio for each chunk
        #     #for i, chunk in enumerate(text_chunks, 1):
        #     #    print(f"Chunk {i}/{len(text_chunks)}: {chunk}")
                
        #         # Generate audio using Kokoro 
        #      #   generator = synthesiser(chunk, voice='af_heart')
                
        #         # Get the first audio output
        #       #  _, _, chunk_audio = next(generator)
                
        #         # Print chunk audio details
        #        # print(f"  Chunk {i} audio details:")
        #        # print(f"    Shape: {chunk_audio.shape}")
        #        # print(f"    Data type: {chunk_audio.dtype}")
        #        # print(f"    Min value: {chunk_audio.min()}")
        #        # print(f"    Max value: {chunk_audio.max()}")
        #        # print(f"    Mean value: {chunk_audio.mean()}")
                
        #         # Add a small pause between chunks (optional)
        #         #pause = np.zeros(int(0.01 * 24000))  # 0.5 second pause
        #         #audio_chunks.append(chunk_audio)
        #         #audio_chunks.append(pause)
            print('hello')    
            from kokoro import KPipeline
            #print('hell0')
            pipeline = KPipeline(lang_code='a')
            
            l2=[]
            
            generator = pipeline(transcription_text, voice='am_adam')
            for j, (gs, ps, audio) in enumerate(generator):
                   l2.append(audio)
            ask=np.concatenate(l2) 

            # Concatenate all audio chunks into a single audio clip
            
            # Save the audio file at 24kHz 
            sf.write(output_path, ask, 24000) 
             
            print(f"Audio generated and saved to: {output_path}") 
             
            # Verify the audio file exists and has content 
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0: 
                print(f"Generated audio file size: {os.path.getsize(output_path)} bytes") 
            else: 
                raise Exception("Generated audio file is empty or doesn't exist") 
             
            # Upload to S3 
            output_audio_s3_url = upload_to_s3(output_path, output_filename, S3_BUCKET) 
            if not output_audio_s3_url: 
                raise Exception("Failed to upload output audio file to S3") 
             
            print(f"Output audio uploaded: {output_audio_s3_url}") 
            tts_end_time = time.time() 
            print(f"TTS processing completed in {tts_end_time - tts_start_time:.2f} seconds") 
             
        except Exception as e: 
            print(f"Error in TTS processing: {str(e)}") 
            traceback.print_exc() 
            # Create a fallback audio file with silence if TTS fails 
            # This ensures the process continues even if TTS fails 
            print("Creating fallback silent audio file...") 
            fallback_audio = np.zeros(24000)  # 1 second of silence at 24kHz 
            output_filename = f"{user_id}_output_{current_epoch_time}.wav" 
            output_path = os.path.join(tempfile.gettempdir(), output_filename) 
            sf.write(output_path, fallback_audio, samplerate=24000) 
            output_audio_s3_url = upload_to_s3(output_path, output_filename, S3_BUCKET) 
            print(f"Fallback silent audio uploaded: {output_audio_s3_url}") 
        
        # Clean up temporary files 
        os.remove(tmp_audio_path) 
        os.remove(output_path)


        total_response_time = time.time() - response_start_time
        metadata = {
            "id": f"{random.randint(1000, 9999)}_{current_epoch_time}",
            "user_id": user_id,
            "inputFile": input_filename,
            "outputFile": output_filename,
            "Transcription": transcription_text,
            "model_used": matching_model["label"] if matching_model else selected_model,
            "duration": current_epoch_time,
            "phrase_text": phrase_text,
            "processing_time": {
                "Transcription": round(transcription_time, 2),
                "total": round(total_response_time, 2)
            }
        }

        if phrase_text is not None:
            metadata.update({
                "phrase_text": phrase_text,
                "input_file": text_filename,
                "word_accuracy": round(word_accuracy, 2) if word_accuracy is not None else None
            })

        # Save metadata
        save_metadata(metadata)

        # Prepare response
        response_data = metadata.copy()
        response_data["message"] = "Process completed successfully"
        
        print("Final response data:", response_data)
        print("=== Audio processing completed ===\n")
        return jsonify(response_data)

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        # Add full traceback for better debugging
        traceback.print_exc()
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500


# ========================================================================================================
# @app.route('/process_audio', methods=['POST'])
# @log_to_file
# def process_audio():
#     global current_model_path, model, processor, pipe, asr_pipeline, synthesiser
    
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file uploaded"}), 400

#     print("\n=== Starting new audio processing request ===")
#     try:
#         # Get basic information
#         user_id = request.form.get('user_id', 'NO_ID')
#         current_epoch_time = int(time.time())

#         # Debug current state
#         print(f"Current model path before processing: {current_model_path}")
#         print("Form data received:", dict(request.form))

#         # Get and validate the selected model
#         selected_model = request.form.get('model')
#         print(f"Selected model from form: {selected_model}")

#         if selected_model:
#             # Find the exact model from AVAILABLE_MODELS
#             matching_model = next((model for model in AVAILABLE_MODELS if model["id"] == selected_model), None)
#             if matching_model:
#                 print(f"Found matching model: {matching_model['id']}")
#                 if matching_model["type"] == "whisper":  # Check if the model type is "whisper"
#                     if selected_model != current_model_path:
#                         print(f"Loading new Whisper model: {selected_model} (current: {current_model_path})")
#                         success, message = load_whisper_model(selected_model)  # Load Whisper model
#                         if success:
#                             current_model_path = selected_model
#                             print(f"Successfully switched to model: {current_model_path}")
#                         else:
#                             print(f"Failed to load Whisper model: {message}")
#                             return jsonify({"error": f"Failed to load model: {message}"}), 500
#                     else:
#                         print("Selected Whisper model is same as current model, no need to reload")

#                 elif matching_model["type"] == "wav2vec2":  # Check if the model type is "wav2vec2"
#                     if selected_model != current_model_path:
#                         print(f"Loading new Wav2Vec2 model: {selected_model} (current: {current_model_path})")
#                         success, message = load_wav2vec2_model(selected_model)  # Load Wav2Vec2 model
#                         if success:
#                             current_model_path = selected_model
#                             print(f"Successfully switched to model: {current_model_path}")
#                         else:
#                             print(f"Failed to load Wav2Vec2 model: {message}")
#                             return jsonify({"error": f"Failed to load model: {message}"}), 500
#                     else:
#                         print("Selected Wav2Vec2 model is same as current model, no need to reload")
#                 else:
#                     print(f"Unknown model type: {matching_model['type']}")
#                     return jsonify({"error": f"Invalid model type: {matching_model['type']}"}), 400
#             else:
#                 print(f"Warning: Selected model '{selected_model}' not found in AVAILABLE_MODELS")
#                 return jsonify({"error": f"Invalid model selection: {selected_model}"}), 400
#         else:
#             print("No model_id provided in form data")
#             selected_model = current_model_path
#             print(f"Using current model: {selected_model}")

#         print(f"Final model being used for processing: {current_model_path}")

#         # Handle text file and reference text
#         phrase_text = None
#         text_filename = None
#         word_accuracy = None

#         print("Files received:", list(request.files.keys()))

#         if 'text_file' in request.files:
#             text_file = request.files['text_file']
#             print(f"Text file received: {text_file.filename}")
#             if text_file.filename != '':
#                 try:
#                     text_filename = secure_filename(text_file.filename)
#                     # Read and decode text content
#                     phrase_text = text_file.read().decode('utf-8').strip()
#                     print(f"Reference text content: '{phrase_text}'")

#                     # Save text file to S3
#                     with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_text_file:
#                         text_file.seek(0)
#                         tmp_text_file.write(text_file.read())
#                         tmp_text_file.flush()
#                         text_s3_filename = f"{user_id}_text_{current_epoch_time}.txt"
#                         text_s3_url = upload_to_s3(tmp_text_file.name, text_s3_filename, S3_BUCKET)
#                         print(f"Text file uploaded to S3: {text_s3_url}")
#                     os.remove(tmp_text_file.name)
#                 except Exception as e:
#                     print(f"Error processing text file: {str(e)}")
#                     return jsonify({"error": f"Error processing text file: {str(e)}"}), 500

#         # Process audio file
#         audio_file = request.files['audio']
#         response_start_time = time.time()

#         # Get file extension and validate format
#         file_extension = os.path.splitext(audio_file.filename)[1].lower()
#         supported_formats = {'.wav', '.mp3', '.m4a', '.mp4', '.weba'}

#         if file_extension not in supported_formats:
#             return jsonify({"error": f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"}), 400

#         # Create a temporary file for the uploaded audio
#         with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_upload_file:
#             audio_file.save(tmp_upload_file.name)
#             uploaded_audio_path = tmp_upload_file.name

#         # Convert to WAV if not already WAV
#         if file_extension != '.wav':
#             try:
#                 print(f"Converting {file_extension} to WAV format...")
#                 if file_extension == '.m4a':
#                     # Special handling for m4a files
#                     audio = AudioSegment.from_file(uploaded_audio_path, format='m4a')
#                 elif file_extension == '.weba':
#                     # Special handling for weba files
#                     audio = AudioSegment.from_file(uploaded_audio_path, format='webm')  # weba is webm audio
#                 else:
#                     audio = AudioSegment.from_file(uploaded_audio_path)
                
#                 # Create temporary WAV file
#                 tmp_audio_path = os.path.join(tempfile.gettempdir(), f"converted_{current_epoch_time}.wav")
#                 audio.export(tmp_audio_path, format='wav')
                
#                 # Remove the original uploaded file
#                 os.remove(uploaded_audio_path)
#             except Exception as e:
#                 os.remove(uploaded_audio_path)
#                 print(f"Error converting audio: {str(e)}")
#                 return jsonify({"error": f"Error converting audio format: {str(e)}"}), 500
#         else:
#             tmp_audio_path = uploaded_audio_path

#         # Preprocess audio to avoid tensor size mismatches
#         try:
#             print("Preprocessing audio file...")
#             import numpy as np
#             import librosa
#             import soundfile as sf
#             import traceback
            
#             # Load and resample audio
#             audio_array, sampling_rate = librosa.load(tmp_audio_path, sr=16000)
#             print(f"Original audio length: {len(audio_array)} samples at {sampling_rate}Hz")
            
#             # Check if audio is too short and pad if necessary
#             min_audio_length = 16000  # 1 second at 16kHz
#             if len(audio_array) < min_audio_length:
#                 print(f"Audio too short ({len(audio_array)} samples), padding to {min_audio_length} samples")
#                 padding = np.zeros(min_audio_length - len(audio_array))
#                 audio_array = np.concatenate([audio_array, padding])
            
#             # Ensure audio length is a multiple of model's expected frame size
#             frame_size = 600  # This is based on your error message
#             remainder = len(audio_array) % frame_size
#             if remainder != 0:
#                 padding_needed = frame_size - remainder
#                 print(f"Padding audio to be multiple of {frame_size} frames (adding {padding_needed} samples)")
#                 padding = np.zeros(padding_needed)
#                 audio_array = np.concatenate([audio_array, padding])
            
#             # Save the preprocessed audio
#             sf.write(tmp_audio_path, audio_array, 16000)
#             print(f"Preprocessed audio length: {len(audio_array)} samples")
            
#         except Exception as e:
#             print(f"Error preprocessing audio: {str(e)}")
#             traceback.print_exc()
#             return jsonify({"error": f"Error preprocessing audio: {str(e)}"}), 500

#         # Save the uploaded audio file temporarily
#         input_filename = f"{user_id}_input_{current_epoch_time}.wav"

#         # Upload input audio file to S3
#         input_audio_s3_url = upload_to_s3(tmp_audio_path, input_filename, S3_BUCKET)
#         if not input_audio_s3_url:
#             raise Exception("Failed to upload input audio file to S3")
        
#         print(f"Input audio uploaded: {input_audio_s3_url}")

#         # Transcription using selected model (Whisper or Wav2Vec2)
#         print("Starting transcription...")
#         transcription_start_time = time.time()

#         try:
#             # Added try-except block around transcription for better error handling
#             if matching_model["type"] == "whisper":
#                 result = pipe(tmp_audio_path, generate_kwargs={"language": "english"})
#             elif matching_model["type"] == "wav2vec2":
#                 result = asr_pipeline(tmp_audio_path)
#             transcription_text = result['text'].strip()
            
#             transcription_time = time.time() - transcription_start_time
#             print(f"Transcription completed in {transcription_time:.2f} seconds")
#             print(f"Transcribed text: '{transcription_text}'")
#         except Exception as e:
#             print(f"Error during transcription: {str(e)}")
#             print(f"Audio file: {audio_file.filename}, preprocessed length: {len(audio_array)}")
#             traceback.print_exc()
#             return jsonify({"error": f"Error during transcription: {str(e)}"}), 500

#         # Post-processing and improved sentence structure
#         transcription_text = format_transcription_for_tts(transcription_text)
#         if not transcription_text.endswith('.'):
#             transcription_text += '.'

#         print(f"Processed Transcription Text: '{transcription_text}'")

#         # Calculate word accuracy if reference text is available
#         if phrase_text:
#             word_accuracy = calculate_word_accuracy(phrase_text, transcription_text)
#             print(f"Word Accuracy: {word_accuracy:.2f}%")

#         # Generate speech using Coqui TTS with improved text segmentation
#         print("Generating speech using Coqui TTS...")
#         tts_start_time = time.time()
#         try:
#             # Check if synthesiser was loaded successfully
#             if synthesiser is None:
#                 raise Exception("TTS model was not loaded properly")
            
#             # Save output audio directly to a temporary file
#             output_filename = f"{user_id}_output_{current_epoch_time}.wav"
#             output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
#             print(f"Processing transcription of length: {len(transcription_text)}")
#             print(f"This is transcription going to tts: {transcription_text}")
            
#             # Split the transcription into proper sentences
#             sentences = split_into_sentences(transcription_text)
#             print(f"Split into {len(sentences)} sentences for TTS processing")
            
#             # Process each sentence separately and concatenate the results
#             audio_segments = []
            
#             for i, sentence in enumerate(sentences):
#                 if not sentence.strip():
#                     continue  # Skip empty sentences
                    
#                 print(f"Processing sentence {i+1}/{len(sentences)}: {sentence}")
                
#                 # Create a temporary file for this sentence
#                 temp_sentence_file = os.path.join(tempfile.gettempdir(), f"sentence_{i}_{current_epoch_time}.wav")
                
#                 # Generate speech for this sentence
#                 synthesiser.tts_to_file(text=sentence, file_path=temp_sentence_file)
                
#                 # Add pause after sentence for natural speech rhythm
#                 if os.path.exists(temp_sentence_file) and os.path.getsize(temp_sentence_file) > 0:
#                     segment = AudioSegment.from_wav(temp_sentence_file)
#                     # Add a short pause (300ms) between sentences
#                     silence = AudioSegment.silent(duration=300)
#                     segment = segment + silence
#                     audio_segments.append(segment)
#                     os.remove(temp_sentence_file)  # Clean up
            
#             # Combine all audio segments
#             if audio_segments:
#                 combined_audio = audio_segments[0]
#                 for segment in audio_segments[1:]:
#                     combined_audio += segment
                
#                 # Export the combined audio
#                 combined_audio.export(output_path, format="wav")
#                 print(f"Combined audio generated and saved to: {output_path}")
#             else:
#                 raise Exception("No audio segments were generated")
            
#             # Verify the audio file exists and has content
#             if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#                 print(f"Generated audio file size: {os.path.getsize(output_path)} bytes")
#             else:
#                 raise Exception("Generated audio file is empty or doesn't exist")
            
#             # Upload to S3
#             output_audio_s3_url = upload_to_s3(output_path, output_filename, S3_BUCKET)
#             if not output_audio_s3_url:
#                 raise Exception("Failed to upload output audio file to S3")
            
#             print(f"Output audio uploaded: {output_audio_s3_url}")
#             tts_end_time = time.time()
#             print(f"TTS processing completed in {tts_end_time - tts_start_time:.2f} seconds")
            
#         except Exception as e:
#             print(f"Error in TTS processing: {str(e)}")
#             traceback.print_exc()
#             # Create a fallback audio file with silence if TTS fails
#             print("Creating fallback silent audio file...")
#             fallback_audio = np.zeros(16000)  # 1 second of silence at 16kHz
#             output_filename = f"{user_id}_output_{current_epoch_time}.wav"
#             output_path = os.path.join(tempfile.gettempdir(), output_filename)
#             sf.write(output_path, fallback_audio, samplerate=16000)
#             output_audio_s3_url = upload_to_s3(output_path, output_filename, S3_BUCKET)
#             print(f"Fallback silent audio uploaded: {output_audio_s3_url}")

#         # Clean up temporary files
#         os.remove(tmp_audio_path)
#         os.remove(output_path)
        
#         total_response_time = time.time() - response_start_time
#         metadata = {
#             "id": f"{random.randint(1000, 9999)}_{current_epoch_time}",
#             "user_id": user_id,
#             "inputFile": input_filename,
#             "outputFile": output_filename,
#             "Transcription": transcription_text,
#             "model_used": matching_model["label"] if matching_model else selected_model,
#             "duration": current_epoch_time,
#             "phrase_text": phrase_text,
#             "processing_time": {
#                 "Transcription": round(transcription_time, 2),
#                 "total": round(total_response_time, 2)
#             }
#         }

#         if phrase_text is not None:
#             metadata.update({
#                 "phrase_text": phrase_text,
#                 "input_file": text_filename,
#                 "word_accuracy": round(word_accuracy, 2) if word_accuracy is not None else None
#             })

#         # Save metadata
#         save_metadata(metadata)

#         # Prepare response
#         response_data = metadata.copy()
#         response_data["message"] = "Process completed successfully"
        
#         print("Final response data:", response_data)
#         print("=== Audio processing completed ===\n")
#         return jsonify(response_data)

#     except Exception as e:
#         print(f"Error processing audio: {str(e)}")
#         # Add full traceback for better debugging
#         traceback.print_exc()
#         return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
#============================================================================================================
# @app.route('/process_audio', methods=['POST'])
# def process_audio():
#     global current_model_path, model, processor, pipe, asr_pipeline
    
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file uploaded"}), 400

#     print("\n=== Starting new audio processing request ===")
#     try:
#         # Get basic information
#         user_id = request.form.get('user_id', 'NO_ID')
#         current_epoch_time = int(time.time())

#         # Debug current state
#         print(f"Current model path before processing: {current_model_path}")
#         print("Form data received:", dict(request.form))

#         # Get and validate the selected model
#         selected_model = request.form.get('model')
#         print(f"Selected model from form: {selected_model}")

#         if selected_model:
#             # Find the exact model from AVAILABLE_MODELS
#             matching_model = next((model for model in AVAILABLE_MODELS if model["id"] == selected_model), None)
#             if matching_model:
#                 print(f"Found matching model: {matching_model['id']}")
#                 if matching_model["type"] == "whisper":  # Check if the model type is "whisper"
#                     if selected_model != current_model_path:
#                         print(f"Loading new Whisper model: {selected_model} (current: {current_model_path})")
#                         success, message = load_whisper_model(selected_model)  # Load Whisper model
#                         if success:
#                             current_model_path = selected_model
#                             print(f"Successfully switched to model: {current_model_path}")
#                         else:
#                             print(f"Failed to load Whisper model: {message}")
#                             return jsonify({"error": f"Failed to load model: {message}"}), 500
#                     else:
#                         print("Selected Whisper model is same as current model, no need to reload")

#                 elif matching_model["type"] == "wav2vec2":  # Check if the model type is "wav2vec2"
#                     if selected_model != current_model_path:
#                         print(f"Loading new Wav2Vec2 model: {selected_model} (current: {current_model_path})")
#                         success, message = load_wav2vec2_model(selected_model)  # Load Wav2Vec2 model
#                         if success:
#                             current_model_path = selected_model
#                             print(f"Successfully switched to model: {current_model_path}")
#                         else:
#                             print(f"Failed to load Wav2Vec2 model: {message}")
#                             return jsonify({"error": f"Failed to load model: {message}"}), 500
#                     else:
#                         print("Selected Wav2Vec2 model is same as current model, no need to reload")
#                 else:
#                     print(f"Unknown model type: {matching_model['type']}")
#                     return jsonify({"error": f"Invalid model type: {matching_model['type']}"}), 400
#             else:
#                 print(f"Warning: Selected model '{selected_model}' not found in AVAILABLE_MODELS")
#                 return jsonify({"error": f"Invalid model selection: {selected_model}"}), 400
#         else:
#             print("No model_id provided in form data")
#             selected_model = current_model_path
#             print(f"Using current model: {selected_model}")

#         print(f"Final model being used for processing: {current_model_path}")

#         # Handle text file and reference text
#         phrase_text = None
#         text_filename = None
#         word_accuracy = None

#         print("Files received:", list(request.files.keys()))

#         if 'text_file' in request.files:
#             text_file = request.files['text_file']
#             print(f"Text file received: {text_file.filename}")
#             if text_file.filename != '':
#                 try:
#                     text_filename = secure_filename(text_file.filename)
#                     # Read and decode text content
#                     phrase_text = text_file.read().decode('utf-8').strip()
#                     print(f"Reference text content: '{phrase_text}'")

#                     # Save text file to S3
#                     with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_text_file:
#                         text_file.seek(0)
#                         tmp_text_file.write(text_file.read())
#                         tmp_text_file.flush()
#                         text_s3_filename = f"{user_id}_text_{current_epoch_time}.txt"
#                         text_s3_url = upload_to_s3(tmp_text_file.name, text_s3_filename, S3_BUCKET)
#                         print(f"Text file uploaded to S3: {text_s3_url}")
#                     os.remove(tmp_text_file.name)
#                 except Exception as e:
#                     print(f"Error processing text file: {str(e)}")
#                     return jsonify({"error": f"Error processing text file: {str(e)}"}), 500

#         # Process audio file
#         audio_file = request.files['audio']
#         response_start_time = time.time()

#         # Save the uploaded audio file temporarily
#         input_filename = f"{user_id}_input_{current_epoch_time}.wav"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
#             tmp_audio_file.write(audio_file.read())
#             tmp_audio_path = tmp_audio_file.name

#         # Upload input audio file to S3
#         input_audio_s3_url = upload_to_s3(tmp_audio_path, input_filename, S3_BUCKET)
#         if not input_audio_s3_url:
#             raise Exception("Failed to upload input audio file to S3")
        
#         print(f"Input audio uploaded: {input_audio_s3_url}")

#         # Transcription using selected model (Whisper or Wav2Vec2)
#         print("Starting transcription...")
#         transcription_start_time = time.time()

#         if matching_model["type"] == "whisper":
#             result = pipe(tmp_audio_path, generate_kwargs={"language": "english"})
#         elif matching_model["type"] == "wav2vec2":
#             result = asr_pipeline(tmp_audio_path)
#         transcription_text = result['text'].strip()
        
#         transcription_time = time.time() - transcription_start_time
#         print(f"Transcription completed in {transcription_time:.2f} seconds")
#         print(f"Transcribed text: '{transcription_text}'")

#         # Post-processing (e.g., removing repeated phrases, correcting grammar)
#         transcription_text = remove_repeated_phrases(transcription_text, n=1)
#         transcription_text = correct_grammar(transcription_text)
#         transcription_text = remove_filler_words(transcription_text)

#         print(f"Processed Transcription Text: '{transcription_text}'")

#         # Calculate word accuracy if reference text is available
#         if phrase_text:
#             word_accuracy = calculate_word_accuracy(phrase_text, transcription_text)
#             print(f"Word Accuracy: {word_accuracy:.2f}%")

#         # Generate speech using Microsoft TTS
#         print("Generating speech using TTS...")
#         tts_start_time = time.time()
#         speech = synthesiser(transcription_text, forward_params={"speaker_embeddings": speaker_embedding})
#         output_filename = f"{user_id}_output_{current_epoch_time}.wav"
#         output_path = os.path.join(tempfile.gettempdir(), output_filename)
#         sf.write(output_path, speech["audio"], samplerate=speech["sampling_rate"])
#         output_audio_s3_url = upload_to_s3(output_path, output_filename, S3_BUCKET)
#         if not output_audio_s3_url:
#             raise Exception("Failed to upload output audio file to S3")
#         print(f"Output audio uploaded: {output_audio_s3_url}")

#         # Clean up temporary files
#         os.remove(tmp_audio_path)
#         os.remove(output_path)

#         total_response_time = time.time() - response_start_time
#         metadata = {
#             "id": f"{random.randint(1000, 9999)}_{current_epoch_time}",
#             "user_id": user_id,
#             "inputFile": input_filename,
#             "outputFile": output_filename,
#             "Transcription": transcription_text,
#             "model_used": selected_model,
#             "duration": current_epoch_time,
#             "phrase_text": phrase_text,
#             "processing_time": {
#                 "Transcription": round(transcription_time, 2),
#                 "total": round(total_response_time, 2)
#             }
#         }

#         if phrase_text is not None:
#             metadata.update({
#                 "phrase_text": phrase_text,
#                 "input_file": text_filename,
#                 "word_accuracy": round(word_accuracy, 2) if word_accuracy is not None else None
#             })

#         # Save metadata
#         save_metadata(metadata)

#         # Prepare response
#         response_data = metadata.copy()
#         response_data["message"] = "Process completed successfully"
        
#         print("Final response data:", response_data)
#         print("=== Audio processing completed ===\n")
#         return jsonify(response_data)

#     except Exception as e:
#         print(f"Error in process_audio: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

#=============================================================================================
# Add this debugging route to verify available models
@app.route('/debug_models', methods=['GET'])
def debug_models():
    return jsonify({
        "available_models": AVAILABLE_MODELS,
        "current_model": current_model_path
    })


#=============================================================================================

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
        
#==============================================================================================
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

#==============================================================================================
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

#==============================================================================================
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
        
#==============================================================================================
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
#==============================================================================================
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
#==============================================================================================
# app.config.update(
#     SESSION_COOKIE_SECURE=True,
#     SESSION_COOKIE_SAMESITE='None',
#     SESSION_COOKIE_DOMAIN='38.188.108.234'  # Your public IP
# )
#==============================================================================================
if __name__ == '__main__':
    app.run(debug=False, host='192.168.0.49', port= 5050)



