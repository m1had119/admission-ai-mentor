import streamlit as st
from google import genai
import os

# ১. পেজ কনফিগারেশন
st.set_page_config(page_title="Admission AI Mentor", page_icon="🎓", layout="wide")

# ২. API Key সেটিং (Streamlit Secrets বা Environment variable থেকে)
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("মেহেরবানি করে Streamlit Secrets-এ GEMINI_API_KEY সেট করুন।")
    st.stop()

client = genai.Client(api_key=api_key)

# ৩. System Prompt লোড করা
@st.cache_data
def load_system_prompt():
    if os.path.exists("system_prompt.txt"):
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "You are an expert admission mentor."

system_prompt = load_system_prompt()

# ৪. Streamlit Session State (মেমোরি সেভ রাখার জন্য)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ৫. ইউজার ইন্টারফেস (UI)
st.title("🎓 Senior Admission Mentor & Strategist")
st.caption("BUET | DU A-Unit | RUET/KUET/CUET | IUT | GST")

# সাইডবারে প্রোগ্রেস কোড ও ট্র্যাকার
with st.sidebar:
    st.header("📌 Progress Tracker")
    user_progress_code = st.text_input("Paste your last PROGRESS_CODE here:")
    if st.button("Load Progress"):
        if user_progress_code:
            st.session_state.chat_history.append({"role": "user", "text": f"{user_progress_code} Ready"})
            st.rerun()

# আগের চ্যাট হিস্ট্রি ডিসপ্লে করা
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# ইউজার ইনপুট বক্সে মেসেজ গ্রহণ
if user_input := st.chat_input("Type 'Ready' or your solution/answer here..."):
    # ইউজারের মেসেজ স্ক্রিনে দেখানো ও স্টেট-এ রাখা
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    # AI এর জন্য প্রম্পট রেডি করা (কনটেক্সট ধরে রাখার জন্য)
    full_prompt = f"System Instructions:\n{system_prompt}\n\nChat History:\n"
    for msg in st.session_state.chat_history:
        full_prompt += f"{msg['role'].capitalize()}: {msg['text']}\n"
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing concepts & generating output..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                response_text = response.text
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "text": response_text})
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
