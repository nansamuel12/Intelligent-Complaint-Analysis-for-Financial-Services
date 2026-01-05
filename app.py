import streamlit as st
import time
from src.rag_pipeline import ComplaintRAG

# Page Configuration
st.set_page_config(
    page_title="Intelligent Complaint Analysis Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize RAG Pipeline (Cached)
@st.cache_resource
def load_rag_pipeline():
    rag = ComplaintRAG()
    try:
        rag.load_resources()
        return rag
    except Exception as e:
        st.error(f"Failed to load RAG resources: {e}")
        return None

rag = load_rag_pipeline()

# Sidebar
with st.sidebar:
    st.title("🤖 Complaint Chatbot")
    st.markdown("""
    This chatbot answers questions about consumer financial complaints using RAG (Retrieval-Augmented Generation).
    
    **Features:**
    - retrieval of relevant complaint chunks
    - LLM-generated answers
    - Source transparency
    """)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main Interface
st.title("Intelligent Complaint Analysis")
st.caption("Ask questions about the complaint database.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Source Evidence"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i+1} (Dist: {source.get('distance', 0):.4f})**")
                    st.text(source.get('chunk_text', 'No text'))
                    st.markdown("---")

# User Input
if prompt := st.chat_input("Ask a question about financial complaints..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Answer
    if rag:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Call RAG Pipeline
                result = rag.answer_question(prompt)
                answer = result.get("answer", "I could not generate an answer.")
                sources = result.get("sources", [])
                
                # Display Answer
                message_placeholder.markdown(answer)
                
                # Display Sources
                if sources:
                    with st.expander("View Source Evidence"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Chunk {i+1} (Dist: {source.get('distance', 0):.4f})**")
                            st.text(source.get('chunk_text', 'No text'))
                            st.markdown("---")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")
    else:
        st.error("RAG pipeline is not initialized. Please check the logs.")
