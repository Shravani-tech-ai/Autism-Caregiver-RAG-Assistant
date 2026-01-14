# scripts/ingest.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PG_DB, PG_USER, PG_PASSWORD, PG_HOST, PG_PORT

from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pypdf import PdfReader

# -------------------------
# Embeddings
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

CONNECTION_STRING = (
    f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

# -------------------------
# Vector Store
# -------------------------
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name="documents",
    embedding_function=embeddings
)

# -------------------------
# Load PDFs
# -------------------------
pdf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/pdfs")
docs = []

for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        reader = PdfReader(os.path.join(pdf_dir, file))
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text) > 200:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file, "page": i}
                    )
                )

print(f"Loaded {len(docs)} chunks")

# -------------------------
# STORE INTO langchain_pg_embedding
# -------------------------
vectorstore.add_documents(docs)

print("✅ Documents embedded and stored via LangChain")
