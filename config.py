from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# PostgreSQL config
PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = int(os.getenv("PG_PORT", 5432))

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
