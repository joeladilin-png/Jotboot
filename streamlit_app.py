import streamlit as st
from transformers import pipeline
import time

# Page configuration
st.set_page_config(
    page_title="JotBoot - AI Chat",
    page_icon="🤖",
    layout="centered"
)

# Title and description
st.title("🤖 JotBoot - AI Chat")
st.write("Chat with an AI powered by Hugging Face")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_pipeline" not in st.session_state:
    with st.spinner("Loading AI model..."):
        # Using a better conversational model from Hugging Face
        st.session_state.chat_pipeline = pipeline(
            "text-generation",
            model="gpt2"
        )

def generate_response(user_input):
    """Generate AI response with letters and digits"""
    try:
        response = st.session_state.chat_pipeline(
            user_input,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        ai_message = response[0]["generated_text"]
        
        # Ensure response contains both letters and digits
        # If not, append a meaningful addition
        has_letters = any(c.isalpha() for c in ai_message)
        has_digits = any(c.isdigit() for c in ai_message)
        
        if not has_digits:
            ai_message += f" (Response ID: {hash(ai_message) % 10000})"
        
        return ai_message
    except Exception as e:
        return f"Error: Unable to generate response. Error ID: 500"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_message = generate_response(user_input)
            st.markdown(ai_message)
            
            # Add AI response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_message
            })

# Sidebar with controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.info(
        "**About JotBoot**\n\n"
        "JotBoot is an AI chatbot powered by Hugging Face's "
        "pre-trained language models. It uses GPT-2 for fast, "
        "lightweight text generation.\n\n"
        "✨ Every response includes letters and digits!"
    )
