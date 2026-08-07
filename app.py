import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Retrieve API Key safely from Streamlit Secrets
API_KEY = st.secrets["GEMINI_API_KEY"]

# Page Configuration
st.set_page_config(page_title="Admission AI Mentor", page_icon="🎓", layout="wide")

st.title("🎓 Admission AI Mentor & Permanent System")
st.caption("Powered by HSC Syllabus, Built-in Admission Database & Custom Gemini API")

# Load System Prompt
@st.cache_data
def load_system_prompt():
    if os.path.exists("system_prompt.txt"):
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "Act as an expert admission mentor."

system_instructions = load_system_prompt()

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    if os.path.exists("faiss_index"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
        st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Render Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Box
user_prompt = st.chat_input("Ask a question, type 'Ready', or submit your 'Answer'...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Context Retrieval
    retrieved_context = "Using internal online database of BUET, DU, IUT, CKRUET, and GST question patterns and syllabus."
    if st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(user_prompt, k=4)
        retrieved_context = "\n\n".join([d.page_content for d in docs])

    # Formulate Final Hybrid Prompt
    full_system_context = f"{system_instructions}\n\n### Primary Source Context:\n{retrieved_context}\n\nNote: Draw questions and concepts seamlessly from standard Bangladeshi Engineering & Varsity Admission Exam archives (BUET/DU/IUT/GST)."
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", full_system_context),
        ("human", "{input}")
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=API_KEY, temperature=0.3)
    chain = prompt_template | llm | StrOutputParser()

    with st.chat_message("assistant"):
        with st.spinner("Thinking & Planning..."):
            response = chain.invoke({"input": user_prompt})
            st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
