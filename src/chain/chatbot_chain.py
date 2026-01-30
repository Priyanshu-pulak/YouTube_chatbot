from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.utils import fetch_transcript, split_transcript
from src.vector_stores import build_qa_vector_store, build_summary_vector_store
from src.chain.qa_chain import build_qa_chain
from src.chain.summary_chain import build_summary_chain
from src.chain.main_chain import build_final_chain

def build_chatbot_chain(video_url: str):
    load_dotenv()

    chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    str_parser = StrOutputParser()

    transcript = fetch_transcript(video_url)
    if not transcript:
        return None

    chunks = split_transcript(transcript)

    qa_store = build_qa_vector_store(chunks, embedding_model)
    qa_chain = build_qa_chain(chat_model, qa_store, str_parser, 4)

    summary_store = build_summary_vector_store(chunks, chat_model, embedding_model)
    summary_chain = build_summary_chain(chat_model, str_parser, summary_store, len(chunks))

    final_chain = build_final_chain(chat_model, qa_chain, summary_chain)

    return final_chain