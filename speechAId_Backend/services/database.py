# services/database.py
import asyncpg
import os
from dotenv import load_dotenv
from fastapi import HTTPException, status

load_dotenv() # Load environment variables from .env file

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "speechaid")
POSTGRES_USER = os.getenv("POSTGRES_USER", "speechaid")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")

pool = None

async def connect_db():
    """Establishes a connection pool to the PostgreSQL database."""
    global pool
    try:
        pool = await asyncpg.create_pool(
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            min_size=1, # Minimum connections in the pool
            max_size=10 # Maximum connections in the pool
        )
        print("PostgreSQL database pool created successfully.")
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        pool = None # Ensure pool is None if connection fails

async def close_db():
    """Closes the PostgreSQL database connection pool."""
    global pool
    if pool:
        print("Closing PostgreSQL database pool.")
        await pool.close()
        pool = None

async def get_db_connection():
    """Dependency to provide a database connection from the pool."""
    if pool is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Database connection pool not initialized.")
    async with pool.acquire() as connection:
        yield connection

