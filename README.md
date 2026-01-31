Voice Enabled RAG AI Chatbot

A voice-enabled Retrieval-Augmented Generation (RAG) chatbot built using Streamlit and LangChain, powered by Groq's LLaMA-3.3-70B model. The chatbot allows users to upload PDF documents, ask questions using voice or text, and receive both text and spoken answers based on document content.

Features

Upload PDF documents
Ask questions using voice or text
RAG-based document question answering
Source document preview with page numbers
Conversational memory
Save and revisit previous chats
Text-to-speech response output
Simple Streamlit interface

Project Structure

voice-rag-ai-chatbot/
│
├── Paper.py
├── requirements.txt
├── README.md

Tech Stack

Python, Streamlit, LangChain, FAISS, HuggingFace Embeddings, Groq LLM API, LLaMA 3.3 (70B), PyPDF2, SpeechRecognition, pyttsx3

How It Works

User uploads a PDF file
Text is extracted from PDF
Text is split into chunks
Embeddings are generated using sentence-transformers
FAISS vector store is created
User asks a question via voice or text
Last 3 chat turns are added as context
Relevant chunks are retrieved from vector store
Prompt is sent to Groq-hosted LLaMA 3.3
Model generates answer using retrieved context
Answer is displayed and spoken aloud
Chat history is stored in session
