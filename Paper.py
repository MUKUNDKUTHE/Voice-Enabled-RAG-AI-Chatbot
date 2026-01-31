import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from secret_api_keys import groq_api_key
import os
import speech_recognition as sr
import pyttsx3 
import threading

os.environ['GROQ_API_KEY'] = groq_api_key

def process_uploaded_file(uploaded_file):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    docs.append({
                        "page_content": chunk,
                        "metadata": {"page_number": i + 1}
                    })
    else:
        st.error("Unsupported file type. Please upload a PDF or CSV file.")
        return None

    texts = [doc["page_content"] for doc in docs]
    metadatas = [doc["metadata"] for doc in docs]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        query = r.recognize_google(audio)
        st.success(f"You said: {query}")
        return query
    except:
        st.error("Sorry, couldn't understand your voice.")
        return ""

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
    )

def context_history(chat):
    history = chat[-3:]
    context = ""
    for i in history:
        context += f"User: {i['question']}\nAssistant: {i['answer']}\n"
    return context

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "saved_chats" not in st.session_state:
        st.session_state.saved_chats = []
    if "selected_chat_index" not in st.session_state:
        st.session_state.selected_chat_index = None
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    page = st.session_state.page

    
    st.title("AI Chatbot")

    with st.sidebar:
        if st.button("Start New Chat"):
            if st.session_state.chat_history:
                st.session_state.saved_chats.append({
                    "history": st.session_state.chat_history.copy(),
                    "file_type": st.session_state.uploaded_file.type if st.session_state.uploaded_file else None
                })
            st.session_state.chat_history = []
            st.session_state.selected_chat_index = None
            st.session_state.uploaded_file = None
            st.session_state.vector_store = None

        if st.session_state.saved_chats:
            options = [f"Chat {i+1}" for i in range(len(st.session_state.saved_chats))]
            index = st.selectbox("View Previous Chats", options=range(len(options)), format_func=lambda x: options[x])
            if st.button("View Selected Chat"):
                st.session_state.selected_chat_index = index
                st.session_state.chat_history = st.session_state.saved_chats[index]["history"]
                st.session_state.uploaded_file = None
                st.session_state.vector_store = None

    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])

    old_chat = st.session_state.selected_chat_index is not None

    if not old_chat:
        if not st.session_state.vector_store:
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if uploaded_file:
                st.session_state.uploaded_file = uploaded_file
                st.session_state.vector_store = process_uploaded_file(uploaded_file)

        if st.session_state.vector_store and st.session_state.uploaded_file:
            llm = load_llm()

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful assistant. Use the context to answer.
If unrelated, say "Out of domain question.If there is any calculation show direct answer no mathematical explaination needed"

Context:
{context}

Question:
{question}

Answer:"""
            )

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            question = ""
            if st.sidebar.button("Speak"):
                question = get_voice_input()
            text_input = st.chat_input("Or type your question here:")
            if text_input:
                question = text_input
            if question:
                with st.chat_message("user"):
                    st.markdown(question)

                with st.spinner("Getting answer..."):
                    ques = f"{context_history(st.session_state.chat_history)} User: {question}"
                    result = qa.invoke({"query": ques})
                    answer = result["result"]
                    sources = result["source_documents"]

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    threading.Thread(target=speak_text, args=(answer,)).start()

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })

                if answer.strip() != "Out of domain question.":
                    if st.session_state.uploaded_file.type == "application/pdf":
                        with st.sidebar:
                            for i, doc in enumerate(sources):
                                page = doc.metadata.get("page_number", "Unknown")
                                preview = " ".join(doc.page_content.split()[:40]) + "..."
                                st.markdown(f"**Source {i+1} (Page {page})**")
                                st.write(preview)
                else:
                    with st.sidebar:
                        st.markdown("**No sources found**")
        else:
            st.info("Upload a PDF to begin.")

if __name__ == "__main__":
    main()
