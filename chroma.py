import streamlit as st

from qdrant_client import QdrantClient

from langchain_qdrant import Qdrant

from langchain_huggingface import HuggingFaceEmbeddings

import os



# --- 1. SECURE CONNECTION ---

# This tries to grab keys from Streamlit Cloud Secrets first.

# If that fails (local laptop), it looks for a .env file.

try:

    qdrant_url = st.secrets["QDRANT_URL"]

    qdrant_api_key = st.secrets["QDRANT_API_KEY"]

except:

    from dotenv import load_dotenv

    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")

    qdrant_api_key = os.getenv("QDRANT_API_KEY")



# --- 2. CONNECT TO CLIENT ---

# Note: We do NOT use recreate_collection here. We just connect.

qdrant_client = QdrantClient(

    url=qdrant_url,

    api_key=qdrant_api_key,

)



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# --- 3. CREATE VECTOR STORE OBJECT ---

# This allows the rest of your app (rag.py, quiz.py) to query the DB

qdrant = Qdrant(

    client=qdrant_client,

    collection_name="Content", 

    embeddings=embeddings,

)



# NO PDF LOADING CODE HERE! 

# The data should already be in the cloud.
