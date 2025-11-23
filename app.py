import streamlit as st
from huggingface_hub import InferenceClient
import os

# ----------------------------------------------------
# Streamlit Setup
# ----------------------------------------------------
st.set_page_config(page_title="HF Chatbot", page_icon="🤖")
st.title("🤖 HuggingFace Llama-3 Chatbot")

# ----------------------------------------------------
# Load Hugging Face API Key (from secrets)
# ----------------------------------------------------
hf_key =  "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"
 # <- Secure secret
if not hf_key:
    st.error("⚠️ Missing Hugging Face API Key! Add it in Streamlit Cloud → Secrets.")
    st.stop()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Choose Model",
        ["meta-llama/Meta-Llama-3-8B-Instruct"],
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------
# Chat History
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------
# HF Chat Function
# ----------------------------------------------------
def hf_chat(prompt: str):
    client = InferenceClient(
        model=model_name,
        token=hf_key
    )

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=300
    )

    return response.choices[0].message["content"]

# ----------------------------------------------------
# Chat Input / UI
# ----------------------------------------------------
if prompt := st.chat_input("Type something..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        reply = hf_chat(prompt)
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
