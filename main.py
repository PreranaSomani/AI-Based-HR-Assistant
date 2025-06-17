# trial_bot_with_faq.py

import os
import ollama
import requests
import logging
import streamlit as st
import shutil
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Globals ---
FILE_PATH = "leavepolicy.txt"
VECTOR_STORE_DIR = "chroma_db"
TIMESTAMP_FILE = "last_modified.txt"

# --- Backend Functions ---

def load_and_chunk_policy(file_path):
    logging.info(f"Loading and chunking policy from {file_path}...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory=VECTOR_STORE_DIR):
    logging.info(f"Creating vector store in {persist_directory}...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory)
    logging.info("Vector store created and persisted.")
    return vector_db

def file_modified(file_path):
    return os.path.getmtime(file_path)

def should_rebuild_vector_db(file_path, timestamp_file):
    if not os.path.exists(timestamp_file):
        return True
    with open(timestamp_file, "r") as f:
        last_timestamp = float(f.read().strip())
    return file_modified(file_path) > last_timestamp

def save_timestamp(file_path, timestamp_file):
    with open(timestamp_file, "w") as f:
        f.write(str(file_modified(file_path)))

def get_or_create_vector_db(file_path=FILE_PATH, persist_dir=VECTOR_STORE_DIR, timestamp_file=TIMESTAMP_FILE, force_rebuild=False):
    if force_rebuild or should_rebuild_vector_db(file_path, timestamp_file):
        logging.info("Detected updated leavepolicy.txt or force rebuild requested.")
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            logging.info(f"Deleted old vector store at {persist_dir}")
        chunks = load_and_chunk_policy(file_path)
        vector_db = create_vector_store(chunks, persist_dir)
        save_timestamp(file_path, timestamp_file)
        return vector_db
    else:
        logging.info(f"Persist directory {persist_dir} found and up to date. Loading existing vector DB...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

def get_ai_response(vector_db, query):
    logging.info(f"Starting similarity search for query: {query}")
    docs = vector_db.similarity_search(query, k=5)
    logging.info(f"Found {len(docs)} documents in the similarity search.")
    
    context = "\n".join([doc.page_content for doc in docs])

    if not context.strip():
        logging.info("No relevant context found for the query.")
        return "🚫 Sorry, I don’t have that information."

    prompt = f"""
You are **NIRA**, an AI-powered **HR policy assistant**. Answer employee queries **clearly and concisely**, only using the information provided in the context below.

📜 **Instructions**:
1. ONLY use the retrieved context to answer the question.
2. DO NOT create emails, letters, or formal requests.
3. DO NOT invent details or assumptions not found in the context.
4. If the context does not contain the answer, say:  
   "Sorry, I don’t have that information."

---

📌 **Question**: {query}

📄 **Relevant Policy Excerpt**:
{context}

---

🧠 Provide a helpful and direct answer below:
"""

    try:
        logging.info(f"Sending request to Ollama model for query: {query}")
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
        logging.info("Received response from Ollama model.")
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error during Ollama API call: {e}")
        return f"❌ Error: {e}"

# --- FastAPI Setup ---

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

vector_db = get_or_create_vector_db()

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query
    logging.info(f"Received query from user: {query}")
    response = get_ai_response(vector_db, query)
    return {"response": response}

# --- Streamlit App ---

def get_response_from_fastapi(query):
    logging.info(f"Calling FastAPI backend with query: {query}")
    response = requests.post("http://localhost:8000/ask", json={"query": query})
    
    if response.status_code == 200:
        logging.info("Received response from FastAPI backend.")
        return response.json()['response']
    else:
        logging.error(f"Error from FastAPI backend: {response.status_code}")
        return "❌ Error calling the backend API."

# --- Streamlit UI Starts Here ---

st.set_page_config(page_title="NIRA - HR Policy Assistant", layout="wide")

st.markdown("<h1 style='font-size: 2.3em;'>🤖 NIRA - Your Smart HR Policy Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2em;'>Welcome to NIRA, your AI-powered assistant for understanding company leave policies. Just ask a question and NIRA will help you out! 🎯</p>", unsafe_allow_html=True)

st.markdown("#### 💡 Frequently Asked Questions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📅 Planned vs Unplanned WFH\n(What’s the difference?)"):
        st.session_state.query_input = "What is the difference between planned and unplanned WFH?"

with col2:
    if st.button("🤱 Maternity Leave\n(Who can avail & duration)"):
        st.session_state.query_input = "What is the maternity leave policy?"

with col3:
    if st.button("🏆 Rewards and recognition\n(Employees nomination?)"):
        st.session_state.query_input = "How are employees nominated for rewards and recognition at AIT Global India?"

st.markdown("#### 📚 Example Questions")
with st.expander("🔍 Click to view some example questions"):
    st.markdown("""
- How many paid leaves am I entitled to annually?
- What is the procedure to apply for sick leave?
- Can I carry forward unused earned or casual leaves to the next year?
- What is the SPOT Award, and who can receive it?
- Is prior approval required for vacation or casual leave?
- Can I apply for emergency leave on short notice?
- What are the consequences of violating the dress code policy?
    """)

st.markdown("<h2 style='margin-top: 20px;'>📝 Ask Your Own Question</h2>", unsafe_allow_html=True)

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

query = st.text_input("💬 Type your question here:", value=st.session_state.query_input, placeholder="e.g., Can I apply for paternity leave during probation period?")

if st.button("🔍 Get Answer"):
    if query.strip():
        with st.spinner("🧠 Thinking... Please wait while NIRA finds the best answer..."):
            response = get_response_from_fastapi(query)
        st.success("✅ NIRA has answered your question!")
        st.markdown(f"**📢 Answer:** {response}")
    else:
        st.warning("❗ Please enter a query.")

st.markdown("---")
st.markdown("🔹 Powered by **LangChain**, **Ollama**, and **Streamlit**  | 🤖 Built with ❤️ for Employee Empowerment")
