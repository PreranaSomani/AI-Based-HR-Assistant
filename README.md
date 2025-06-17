# NIRA – AI-Based HR Assistant using RAG

NIRA is an open-source HR assistant that answers employee queries using Retrieval-Augmented Generation (RAG). It semantically retrieves relevant HR policy content and generates accurate, grounded responses using a local LLM.

## Features

- RAG-powered factual answering (no hallucinations)  
- Built with LangChain, HuggingFace, ChromaDB, Qwen (via Ollama)  
- FastAPI backend and Streamlit frontend  
- Average response time under 5 seconds  
- Supports up to 20 concurrent users  

## How It Works

1. HR policies are chunked and embedded using MiniLM  
2. Stored in ChromaDB for semantic search  
3. User queries retrieve top-matching chunks  
4. Qwen LLM generates accurate answers  
5. FastAPI handles logic; Streamlit provides user interface  

## Tech Stack

LangChain, ChromaDB, Qwen 2.5, Ollama, FastAPI, Streamlit

## Author

Prerana Somani  
Final-year ICT student at PDEU | Incoming MS Data Science @ Stony Brook  
[LinkedIn](https://www.linkedin.com/in/your-profile)  
[GitHub](https://github.com/your-username)

