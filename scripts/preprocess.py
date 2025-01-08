import re
from pymongo import MongoClient
from pymongo.errors import WriteError

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# MongoDB Connection Setup
mongo_client = MongoClient("mongodb://localhost:27017/")  # Replace with your connection string
articles_db = mongo_client["dataset"]  # Replace with your database name
speeches_db = mongo_client["prime_minister"]

# Load data from MongoDB collections
articles_collection = articles_db["news_articles"]  # Replace with your articles collection name
speeches_collection = speeches_db["articles_speeches"]  # Replace with your speeches collection name

# Track errors
error_count = 0

# Preprocessing for Articles Database
for document in articles_collection.find({}, {"_id": 1, "text": 1}):
    try:
        processed_text = preprocess_text(document["text"])
        articles_collection.update_one({"_id": document["_id"]}, {"$set": {"processed_text": processed_text}})
    except WriteError as e:
        print(f"Error processing document with _id: {document['_id']}. Skipping.")
        error_count += 1

# Preprocessing for Speeches Database
for document in speeches_collection.find({}, {"_id": 1, "article_text": 1}):
    try:
        processed_text = preprocess_text(document["article_text"])
        speeches_collection.update_one({"_id": document["_id"]}, {"$set": {"processed_text": processed_text}})
    except WriteError as e:
        print(f"Error processing document with _id: {document['_id']}. Skipping.")
        error_count += 1

print(f"Preprocessing complete. 'processed_text' added to all documents. Total errors encountered: {error_count}")
