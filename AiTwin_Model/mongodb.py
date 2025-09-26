# mongodb.py - MongoDB Connection Setup (Conceptual)
import motor.motor_asyncio
from typing import AsyncGenerator
import os

# Environment variables for MongoDB Atlas connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://bhuvangs2004:Hh1skD9CcVo0N0gc@aitwin.hgotapz.mongodb.net/?retryWrites=true&w=majority&appName=aitwin")
DATABASE_NAME = os.environ.get("MONGO_DB_NAME", "chatdb")

# Global variables for client and database instance
client: motor.motor_asyncio.AsyncIOMotorClient = None
database: motor.motor_asyncio.AsyncIOMotorDatabase = None

async def connect_to_mongo():
    """Initializes the MongoDB connection."""
    global client, database
    try:
        print("Connecting to MongoDB Atlas...")
        client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000  # 5 second timeout
        )
        database = client[DATABASE_NAME]
        await client.admin.command('ping')
        print(f"✅ Successfully connected to MongoDB Database: {DATABASE_NAME}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        # In a real app, you might raise an error here or exit
        raise

async def close_mongo_connection():
    """Closes the MongoDB connection."""
    global client
    if client is not None:
        client.close()
        print("🗑️ MongoDB connection closed.")

async def get_mongo_db() -> AsyncGenerator[motor.motor_asyncio.AsyncIOMotorDatabase, None]:
    """Dependency for FastAPI endpoints to get the database instance."""
    if database is not None:
        yield database
    else:
        # Fallback if connection fails, though connect_to_mongo should run on startup
        raise Exception("MongoDB connection not established.")

# We will need to call connect_to_mongo() on FastAPI startup (in agent.py)