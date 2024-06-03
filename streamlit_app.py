"""
This module creates a Streamlit-based Llama2 chatbot. It initializes necessary
components, loads a PDF file, processes it, and sets up a chat interface for user interaction.
"""

import os
import streamlit as st
from loguru import logger

from src.managers.chromadb_manager import ChromaDBManager
from src.managers.load_chunk_manager import LoadingChunkingManager
from src.managers.llm_manager import LLMManager

st.title("Welcome to your personal Llama2 chatbot!")

# Set a default model
if "llama_model" not in st.session_state:
    st.session_state["llama_model"] = "llama2_model"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize managers and components
if "initialized" not in st.session_state:
    logger.info("Preparing assistant to chat...")

    file_path = f"C:{os.sep}Users{os.sep}Jose{os.sep}Documents{os.sep}COVER LETTER IKEA.pdf"
    
    st.session_state.lang_chain_manager = LoadingChunkingManager()
    st.session_state.chroma_db_manager = ChromaDBManager()
    st.session_state.llm_manager = LLMManager()
    
    data = st.session_state.lang_chain_manager.load_pdf(file_path)
    chunks = st.session_state.lang_chain_manager.create_chunks(data)
    vector_db = st.session_state.chroma_db_manager.add_vector_to_db(chunks)
    
    llm_llama2 = st.session_state.llm_manager.load_llama2()
    st.session_state.retriever = st.session_state.llm_manager.get_retriever(llm_llama2, vector_db)
    st.session_state.prompt = st.session_state.llm_manager.get_chat_prompt_template()
    st.session_state.chain = st.session_state.llm_manager.get_chain(st.session_state.retriever, st.session_state.prompt, llm_llama2)
    
    st.session_state.initialized = True
    logger.info("Process finished")
    logger.info("Launching chatbot...")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type a message"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    assistant_response = st.session_state.chain.invoke(str(prompt))
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    # Display assistant message in chat message container
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
