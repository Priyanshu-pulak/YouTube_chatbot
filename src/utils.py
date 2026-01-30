from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter

def format_docs(retrieved_docs):
    return "\n".join(doc.page_content for doc in retrieved_docs)

def get_video_id(url: str) -> str:
        if "v=" in url:
            part = url.split("v=")[1]
            return part.split("&")[0]
        return None

def fetch_transcript(video_url: str) -> str:
    video_id = get_video_id(video_url)
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        print("Transcript fetched successfully!")
    except TranscriptsDisabled:
        transcript = ""
        print("Transcripts disabled for this video.")
    return transcript

def split_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )

    chunks = splitter.create_documents([transcript])
    return chunks