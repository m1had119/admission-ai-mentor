import os
import time
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(page_title="Admission AI Mentor", page_icon="🎓", layout="wide")

st.title("🎓 Admission AI Mentor & Hybrid Knowledge System")
st.caption("Powered by HSC Syllabus, PDF Knowledge Base & Online Admission Question Banks (BUET/DU/IUT/GST)")

# Sidebar for Setup & Uploads
with st.sidebar:
    st.header("⚙️ Settings & Knowledge Base")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    st.subheader("📚 PDF Question Banks / Books")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    process_btn = st.button("Process & Load PDF Knowledge Base")

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

# Process Massive PDFs in Batches safely
if process_btn and uploaded_files:
    if not api_key:
        st.error("Please enter a valid Gemini API Key first!")
    else:
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        all_text = ""
        total_files = len(uploaded_files)
        
        for idx, pdf in enumerate(uploaded_files):
            status_box.text(f"Extracting text from PDF {idx+1}/{total_files}: {pdf.name}")
            try:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            except Exception as e:
                st.warning(f"Skipped corrupted/encrypted file: {pdf.name}")
            
            progress_bar.progress((idx + 1) / total_files)

        if all_text.strip():
            status_box.text("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_text(all_text)
            
            status_box.text(f"Generating Vector Embeddings for {len(chunks)} text chunks in batches...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            
            # Batching to avoid Rate Limit error
            batch_size = 50
            vector_store = None
            
            chunk_progress = st.progress(0)
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings)
                else:
                    vector_store.add_texts(batch_chunks)
                
                chunk_progress.progress(min((i + batch_size) / len(chunks), 1.0))
                time.sleep(1)  # 1 second delay to respect API quota limits
            
            st.session_state.vector_store = vector_store
            status_box.empty()
            st.success("✅ All PDF Knowledge Base Successfully Processed & Indexed!")
        else:
            st.error("❌ No readable text found in uploaded PDFs!")

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

        # Context Retrieval
        retrieved_context = "No PDF context uploaded. Use internal online database of BUET, DU, IUT, CKRUET, and GST question patterns."
        if st.session_state.vector_store:
            docs = st.session_state.vector_store.similarity_search(user_prompt, k=4)
            retrieved_context = "\n\n".join([d.page_content for d in docs])

        # Formulate Final Hybrid Prompt
        full_system_context = f"{system_instructions}\n\n### Primary Source Context (PDFs/Online Admission DB):\n{retrieved_context}\n\nNote: If PDF context is insufficient or missing, seamlessly draw questions and concepts from standard Bangladeshi Engineering & Varsity Admission Exam archives (BUET/DU/IUT/GST)."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", full_system_context),
            ("human", "{input}")
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.3)
        chain = prompt_template | llm | StrOutputParser()

        with st.chat_message("assistant"):
            with st.spinner("Thinking & Retrieving Admission Questions..."):
                response = chain.invoke({"input": user_prompt})
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
