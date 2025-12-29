from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from app.core.config import MONGO_URI, DB_NAME

# MongoDB Atlas client (official recommended way)
client = MongoClient(
    MONGO_URI,
    server_api=ServerApi('1')
)

# Ping on startup (optional but useful)
try:
    client.admin.command('ping')
    print(" Connected to MongoDB Atlas successfully")
except Exception as e:
    print(" MongoDB connection error:", e)

db = client[DB_NAME]

users_collection = db["users"]
voice_records_collection = db["voice_records"]
