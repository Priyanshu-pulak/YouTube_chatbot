from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.utils import fetch_transcript, split_transcript
from src.vector_stores.qa_vector_store import build_qa_vector_store
from src.vector_stores.summary_vector_store import build_summary_vector_store
from src.chain.qa_chain import build_qa_chain
from src.chain.summary_chain import build_summary_chain
from src.chain.main_chain import build_final_chain
    
def youtube_chatbot(video_url: str = "https://www.youtube.com/watch?v=XmRrGzR6udg&list=PLgUwDviBIf0rAuz8tVcM0AymmhTRsfaLU&index=6"):
    load_dotenv()

    chat_model = ChatGoogleGenerativeAI (model = 'gemini-2.5-flash-lite')
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    str_parser = StrOutputParser()

    transcript = fetch_transcript(video_url)
    if not transcript:
        return

    chunks = split_transcript(transcript)

    qa_store = build_qa_vector_store(chunks, embedding_model)
    qa_chain = build_qa_chain(chat_model, qa_store, str_parser, 4)

    summary_store = build_summary_vector_store(chunks, chat_model, embedding_model)
    summary_chain = build_summary_chain(chat_model, str_parser, summary_store, len(chunks))

    final_chain = build_final_chain(chat_model, qa_chain, summary_chain)

    print("YouTube Video Q&A System")
    print("Ask questions about the video. Type 'quit' or 'exit' to stop.\n")
    print("-" * 63)

    while True:
        question = input("\nYour question: \n").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        result = final_chain.invoke(question)
        print(f'\n{result}')

if __name__ == "__main__":
    youtube_chatbot()