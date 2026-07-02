import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Client setup
gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Model & chunk constants
CHAT_MODEL = "gemini-flash-latest"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
