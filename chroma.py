import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "Content"


@st.cache_resource(show_spinner="Connecting to knowledge base...")
def get_qdrant_client():
    client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        timeout=30,
    )

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

    return client


@st.cache_resource(show_spinner="Loading AI knowledge model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


qdrant_client = get_qdrant_client()
embeddings = get_embeddings()

qdrant = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)
