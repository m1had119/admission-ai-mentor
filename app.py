import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Configuration
st.set_page_config(page_title="Admission AI Mentor", page_icon="🎓", layout="wide")

st.title("🎓 Admission AI Mentor & RAG System")
st.caption("Custom-trained with HSC & Admission Syllabus, Vector DB, and Gemini API")

# Sidebar for Setup & Uploads
with st.sidebar:
    st.header("⚙️ Settings & Knowledge Base")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    st.subheader("📚 PDF Question Banks / Books")
    uploaded_files = st.file_uploader("Upload PDFs (Question Banks/Notes)", type=["pdf"], accept_multiple_files=True)
    
    process_btn = st.button("Process & Load Knowledge Base")

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Load System Prompt
@st.cache_data
def load_system_prompt():
    if os.path.exists("system_prompt.txt"):
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "Act as an expert admission mentor."

system_instructions = load_system_prompt()

# Process PDFs
if process_btn and uploaded_files:
    if not api_key:
        st.error("Please enter a valid Gemini API Key first!")
    else:
        with st.spinner("Extracting & Indexing PDFs... Please wait"):
            all_text = ""
            for pdf in uploaded_files:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            
            # Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_text(all_text)
            
            # Create Vector Database using Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            st.success("✅ Knowledge Base Successfully Created!")

# Render Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Box
user_prompt = st.chat_input("Ask a question, type 'Ready', or submit your 'Answer'...")

if user_prompt:
    if not api_key:
        st.error("⚠️ Please insert your Gemini API Key in the sidebar to start!")
    else:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Context Retrieval from Vector Store
        retrieved_context = ""
        if st.session_state.vector_store:
            docs = st.session_state.vector_store.similarity_search(user_prompt, k=3)
            retrieved_context = "\n\n".join([d.page_content for d in docs])

        # Formulate Final Prompt
        full_system_context = f"{system_instructions}\n\n### Relevant PDF Context:\n{retrieved_context}"
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", full_system_context),
            ("human", "{input}")
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.3)
        chain = prompt_template | llm | StrOutputParser()

        with st.chat_message("assistant"):
            with st.spinner("Thinking & Planning..."):
                response = chain.invoke({"input": user_prompt})
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
  
