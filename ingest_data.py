# ingest_data.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
import glob
import os
from dotenv import load_dotenv

load_dotenv()

# Connect using local .env
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# THIS IS THE LINE THAT WIPES AND RESETS (Good for ingestion script only)
print("🧹 Clearing and Recreating Database Collection...")
client.recreate_collection(
    collection_name="Content",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Qdrant(client=client, collection_name="Content", embeddings=embeddings)

# Load PDFs
print("📂 Loading PDFs...")
pdf_files = glob.glob("PDFs/*.pdf")
for pdf_file in pdf_files:
    print(f"   Processing: {pdf_file}")
    documents = PyPDFLoader(file_path=pdf_file).load()
    vector_store.add_documents(documents)

print("✅ Data Uploaded Successfully to Cloud!")
