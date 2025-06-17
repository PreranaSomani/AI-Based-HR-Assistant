# AI-Based-HR-Assistant
An AI-based HR Assistant using Retrieval-Augmented Generation (RAG) for accurate, real-time responses to HR queries. Built with LangChain, HuggingFace, ChromaDB, Qwen, FastAPI, and Streamlit.

# NIRA – AI-Based HR Assistant using RAG

NIRA (Natural Interaction for Responsive Assistance) is an AI-powered HR assistant that uses **Retrieval-Augmented Generation (RAG)** to answer employee queries with contextual accuracy and real-time responsiveness. It is designed to automate routine HR interactions and reduce manual workload, using only open-source tools and local deployment.

---

## 💡 Problem Statement

Employees often face delays and confusion when accessing HR policy information. Traditional portals lack conversational interfaces and context-aware responses. Manual support is time-consuming and inconsistent.

---

## 🚀 Solution Overview

NIRA uses a combination of semantic document retrieval and language model generation to:

- Accurately respond to employee HR-related queries (e.g., leave policy, benefits)
- Retrieve information from internal documents using **vector-based search**
- Ensure answers are grounded in actual HR documents (no hallucination)

---

## 🎯 Project Objectives

- ✅ **Accurate Query Handling** using RAG for factual answers  
- ✅ **Open-Source Development** using LangChain, ChromaDB, Qwen, etc.  
- ✅ **User-Friendly Interface** with Streamlit and FastAPI  
- ✅ **Scalable & Efficient** system performance for real-world use  

---

## 🧠 Tech Stack

- **LangChain** – for document chunking & pipeline orchestration  
- **HuggingFace Transformers** – MiniLM embeddings for vector search  
- **ChromaDB** – lightweight vector store for semantic retrieval  
- **Qwen 2.5 LLM (via Ollama)** – for grounded, context-based answer generation  
- **FastAPI** – backend for processing user queries  
- **Streamlit** – interactive web frontend for users  
- **n8n (optional)** – automation and bot integration (e.g., Telegram)

---

## 🛠️ System Architecture

1. **Document Ingestion** – HR documents chunked with context preservation  
2. **Embedding & Storage** – Text embedded using MiniLM, stored in ChromaDB  
3. **Query Processing** – User query embedded and matched with relevant chunks  
4. **Answer Generation** – Qwen LLM responds using retrieved context only  
5. **API & UI** – FastAPI serves queries; Streamlit offers simple UX  

---

## 📈 Results & Performance

- 💬 Answers common HR queries (e.g., maternity leave) with 90%+ accuracy  
- ⏱️ Average response time < 5 seconds  
- 🧠 No hallucinations – responses grounded in actual documents  
- 🧪 Handles 20+ concurrent requests reliably  

---

## 🧩 Future Enhancements

- 📚 Support for multiple HR documents  
- 🗣️ Voice-to-text and natural language input  
- 🌐 Multilingual support  
- 📎 Integration with platforms like Slack or MS Teams  
- 🧠 Fine-tuning for organization-specific needs  

---

## 👩‍💼 Use Case

This project demonstrates how **AI can be effectively applied in HR systems** to boost efficiency, accuracy, and employee experience, using cost-effective, local, open-source infrastructure.

---

## 👩‍💻 Author

**Prerana Somani**  
Final-year B.Tech ICT student, Pandit Deendayal Energy University  
✨ Incoming MS in Data Science @ Stony Brook University  
📬 [LinkedIn](https://www.linkedin.com/in/your-profile) • [GitHub](https://github.com/your-username)

---

## 📜 License

MIT License – feel free to use, adapt, and build upon this project!

