import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    if "v=" in url:
        part = url.split("v=")[1]
        return part.split("&")[0]
    return None

# Page configuration
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

# Sidebar for video input
with st.sidebar:
    st.header("📹 Video Setup")
    
    # Video URL input
    video_input = st.text_input(
        "YouTube Video URL or ID",
        placeholder="e.g., https://youtube.com/watch?v=Gfr50f6ZBvo or Gfr50f6ZBvo",
        help="Enter the full YouTube URL or just the video ID"
    )
    
    # Process button
    if st.button("Process Video", type="primary"):
        if video_input:
            # Extract video ID from URL or use as-is if it's already an ID
            if "youtube.com" in video_input or "youtu.be" in video_input:
                video_id = get_video_id(video_input)
                if not video_id:
                    st.error("❌ Could not extract video ID from URL. Please check the URL format.")
                    st.stop()
            else:
                video_id = video_input.strip()
            
            with st.spinner("Processing video transcript..."):
                try:
                    # Initialize models
                    chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
                    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    parser = StrOutputParser()
                    
                    # Fetch transcript
                    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
                    transcript = " ".join(chunk.text for chunk in transcript_list)
                    
                    if not transcript:
                        st.error("No transcript found for this video.")
                        st.stop()
                    
                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                    )
                    chunks = splitter.create_documents([transcript])
                    
                    # Create vector store
                    vector_store = FAISS.from_documents(
                        documents=chunks,
                        embedding=embedding_model
                    )
                    
                    # Create retriever
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    
                    # Create prompt template
                    prompt = PromptTemplate(
                        template="""
                        You are a helpful assistant.
                        Answer only from the provided transcript context.
                        If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.
                        Context: {context}
                        Question: {question}
                        """,
                        input_variables=["context", "question"]
                    )
                    
                    # Format docs function
                    def format_docs(retrieved_docs):
                        context_text = ""
                        for doc in retrieved_docs:
                            context_text += doc.page_content + "\n"
                        return context_text
                    
                    # Create chain
                    parallel_chain = RunnableParallel({
                        'context': retriever | RunnableLambda(format_docs),
                        'question': RunnablePassthrough()
                    })
                    
                    main_chain = parallel_chain | prompt | chat_model | parser
                    
                    # Store in session state
                    st.session_state.chain = main_chain
                    st.session_state.video_processed = True
                    st.session_state.current_video_id = video_id
                    st.session_state.chat_history = []
                    
                    st.success("✅ Video processed successfully! You can now ask questions.")
                    
                except TranscriptsDisabled:
                    st.error("❌ Transcripts are disabled for this video.")
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
        else:
            st.warning("Please enter a YouTube video URL or ID.")
    
    # Display current video info
    if st.session_state.video_processed:
        st.success(f"✅ Video ID: {st.session_state.current_video_id}")
        if st.button("Clear Video"):
            st.session_state.video_processed = False
            st.session_state.chain = None
            st.session_state.chat_history = []
            st.rerun()

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
            placeholder="e.g. Gie me a summary of the video."
        )
        submit_button = st.form_submit_button("Ask", type="primary")
        
        if submit_button and question.strip():
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.chain.invoke(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, result))
                    
                    # Display the new answer
                    st.markdown(f"**🙋 Question:** {question}")
                    st.markdown(f"**🤖 Answer:** {result}")
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    # Clear chat history button
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.info("👈 Please enter a YouTube video URL or ID in the sidebar and click 'Process Video' to get started.")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Get a YouTube video**: Copy the full URL from YouTube (e.g., `https://youtube.com/watch?v=Gfr50f6ZBvo`) or just the video ID (`Gfr50f6ZBvo`)
    2. **Paste the URL or ID** in the sidebar
    3. **Click 'Process Video'** to fetch and process the transcript
    4. **Ask questions** about the video content in the chat interface
    5. **Get AI-powered answers** based on the video transcript
    
    **Note**: Make sure you have your `GOOGLE_API_KEY` set in your environment variables or `.env` file.
    """)

# Environment check
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("⚠️ GOOGLE_API_KEY not found in environment variables. Please set it to use this app.")
