from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-lite')
embedding_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
parser = StrOutputParser()

def get_video_id(url: str) -> str:
    if "v=" in url:
        part = url.split("v=")[1]
        return part.split("&")[0]
    return None

video_id = get_video_id("https://www.youtube.com/watch?v=Nq7ok-OyEpg&list=PLgUwDviBIf0rAuz8tVcM0AymmhTRsfaLU&index=4&t=2535s") 
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages = ['en'])
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    transcript = ""
    print("Transcripts disabled for this video.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)

chunks = splitter.create_documents([transcript])
vector_store = FAISS.from_documents(
    documents = chunks,
    embedding = embedding_model
)

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 4}
)

prompt = PromptTemplate(
    template = """
        You are a helpful assistant.
        Answer only from the provided transcript context.
        If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.
        Context: {context}
        Question: {question}
    """,
    input_variables = ["context", "question"]
)

def format_docs(retrieved_docs):
    context_text = ""
    for doc in retrieved_docs:
        context_text += doc.page_content + "\n"
    return context_text
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | chat_model | parser

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
    
    result = main_chain.invoke(question)
    print(f'\n{result}')