# scripts/query.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PG_DB, PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, GEMINI_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# Embeddings (CORRECT)
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = PGVector(
    connection_string=f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}",
    collection_name="documents",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------
# LLM (Gemini)
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2
)

docs = retriever.invoke("what is autism")
print("\nRETRIEVED DOCS COUNT:", len(docs))
if docs:
    print("\nSAMPLE DOC:\n", docs[0].page_content[:500])

# -------------------------
# Prompt
# -------------------------
prompt = ChatPromptTemplate.from_template("""
You are an assistant for autism caregivers.
Use ONLY the context below. If the context does not answer the question, say "I do not know".

Context:
{context}

Question:
{question}

Answer:
""")

# -------------------------
# LCEL RAG PIPELINE
# -------------------------
rag_pipeline = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------
# CLI LOOP
# -------------------------
while True:
    question = input("\nAsk a caregiver question (or 'exit'): ")
    if question.lower() in ["exit", "quit"]:
        break

    answer = rag_pipeline.invoke(question)
    print("\nAnswer:\n", answer)
