import streamlit as st
from dotenv import load_dotenv
import os

from src.chain.chatbot_chain import build_chatbot_chain

load_dotenv()

st.set_page_config(
    page_title="YouTube Q&A Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube Video Q&A Assistant")
st.markdown("Ask questions about any YouTube video with transcripts!")

# Initialize session state
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_video_url' not in st.session_state:
    st.session_state.current_video_url = None

with st.sidebar:
    st.header("📹 Video Setup")
    
    video_input = st.text_input(
        "YouTube Video URL or ID",
        placeholder = "e.g., https://youtube.com/watch?v=Gfr50f6ZBvo or Gfr50f6ZBvo",
        help = "Enter the full YouTube URL or just the video ID"
    )
    
    # Process button
    if st.button("Process Video", type = "primary"):
        if video_input:
            video_url = video_input

            with st.spinner("Processing video transcript..."):
                try:
                    final_chain = build_chatbot_chain(video_url)
                    
                    if not final_chain:
                        st.error("❌ No transcript found for this video or failed to process.")
                        st.stop()
                    
                    # Store in session state
                    st.session_state.chain = final_chain
                    st.session_state.video_processed = True
                    st.session_state.current_video_url = video_url
                    st.session_state.chat_history = []
                    
                    st.success("✅ Video processed successfully! You can now ask questions.")
                    
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
        else:
            st.warning("Please enter a YouTube video URL or ID.")
    
    # Display current video info
    if st.session_state.video_processed:
        st.success(f"✅ Video ready")

# Main chat interface
if st.session_state.video_processed and st.session_state.chain:
    st.header("💬 Ask Questions")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 Question {i+1}:** {question}")
            st.markdown(f"**🤖 Answer:** {answer}")
            st.divider()
    
    # Question input
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question:",
            placeholder="e.g., Give me a summary of the video or What is discussed in the video?"
        )
        submit_button = st.form_submit_button("Ask", type="primary")
        
        if submit_button and question.strip():
            with st.spinner("🤔 Generating answer..."):
                try:
                    result = st.session_state.chain.invoke(question)
                    
                    output_text = result.get('output', str(result)) if isinstance(result, dict) else str(result)
                    
                    st.session_state.chat_history.append((question, output_text))
                    
                    st.markdown(f"**🙋 Question:** {question}")
                    st.markdown(f"**🤖 Answer:** {output_text}")
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.info("👈 Please enter a YouTube video URL or ID in the sidebar and click 'Process Video' to get started.")

with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Get a YouTube video**: Copy the full URL from YouTube (e.g., `https://youtube.com/watch?v=Gfr50f6ZBvo`) or just the video ID (`Gfr50f6ZBvo`)
    2. **Paste the URL or ID** in the sidebar
    3. **Click 'Process Video'** to fetch and process the transcript
    4. **Ask questions** about the video content in the chat interface
    5. **Get AI-powered answers** based on the video transcript
    
    **Note**: Make sure you have your `GOOGLE_API_KEY` set in your environment variables or `.env` file.
    """)

if not os.getenv("GOOGLE_API_KEY"):
    st.warning("⚠️ GOOGLE_API_KEY not found in environment variables. Please set it to use this app.")
