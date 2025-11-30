import streamlit as st
import os
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. SECURE CONNECTION ---
try:
    qdrant_url = st.secrets["QDRANT_URL"]
    qdrant_api_key = st.secrets["QDRANT_API_KEY"]
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. CONNECT TO CLIENT ---
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# --- 3. DEFINE EMBEDDINGS (GOOGLE) ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# --- 4. CONNECT TO VECTOR STORE ---
# Ensure "vector_db_data" matches your actual Qdrant Cloud collection name!
qdrant = Qdrant(
    client=qdrant_client,
    collection_name="vector_db_data", 
    embeddings=embeddings,
)
