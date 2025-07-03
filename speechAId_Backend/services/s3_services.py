# services/s3_service.py
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
from typing import Optional
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() # Load environment variables from .env file

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION') # Updated variable name
S3_BUCKET = os.getenv('S3_BUCKET') # Updated variable name

s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_REGION and S3_BUCKET:
    try:
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION, # Use new variable name
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print(f"S3 client initialized successfully for bucket: {S3_BUCKET}")
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
else:
    print("AWS S3 credentials or bucket name not fully set. S3 upload functionality will be disabled.")


# Helper function to upload file to S3
async def upload_file_to_s3(file_content: bytes, bucket_name: str, object_name: str, content_type: str) -> Optional[str]:
    """Uploads a file to an S3 bucket."""
    if s3_client is None:
        raise NoCredentialsError("S3 client not initialized.")
    try:
        # boto3 expects a file-like object or bytes for upload_fileobj
        s3_client.upload_fileobj(
            FileContent(file_content), # Use our custom FileContent helper
            bucket_name,
            object_name,
            ExtraArgs={'ContentType': content_type}
        )
        s3_url = f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com/{object_name}" # Use AWS_REGION
        return s3_url
    except ClientError as e:
        print(f"S3 Client Error during upload: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during S3 upload: {e}")
        raise

# Helper class to mimic file-like object for boto3.upload_fileobj
# This is needed because FastAPI's UploadFile directly exposes .read() but might not be seekable
# or directly compatible with boto3's internal expectations without this wrapper.
class FileContent:
    def __init__(self, content: bytes):
        self._content = content
        self._position = 0

    def read(self, size: Optional[int] = -1) -> bytes:
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

