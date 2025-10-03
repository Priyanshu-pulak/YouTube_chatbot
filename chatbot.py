from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import time
from langchain.chains.summarize import load_summarize_chain
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-lite')
embedding_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
str_parser = StrOutputParser()

def get_video_id(url: str) -> str:
    if "v=" in url:
        part = url.split("v=")[1]
        return part.split("&")[0]
    return None

video_id = get_video_id("https://www.youtube.com/watch?v=XmRrGzR6udg&list=PLgUwDviBIf0rAuz8tVcM0AymmhTRsfaLU&index=6") 
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

qa_vector_store = FAISS.from_documents(
    documents = chunks,
    embedding = embedding_model
)

qa_retriever = qa_vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 4}
)

qa_prompt = PromptTemplate(
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
    return "\n".join(doc.page_content for doc in retrieved_docs)

qa_parallel_chain = RunnableParallel({
    'context': qa_retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

qa_chain = qa_parallel_chain | qa_prompt | chat_model | str_parser

chunk_summary_chain = load_summarize_chain(chat_model, chain_type='stuff')
final_summaries = []

for i, chunk in enumerate(chunks):

    print(f"Summarizing chunk {i + 1}/{len(chunks)}:")
    
    summary_output = chunk_summary_chain.invoke([chunk])
    summary_text = summary_output['output_text']
    
    final_summaries.append(Document(page_content = summary_text))
    
    time.sleep(8)

print("All chunks summarized successfully!")

summary_vector_store = FAISS.from_documents(
    documents = final_summaries,
    embedding = embedding_model
)

summary_retriever = summary_vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": len(final_summaries)}
)

summary_prompt = PromptTemplate(
    template = """
        You are a helpful assistant.
        Summarize the following transcript into the manner user specified otherwise in a paragraph.
        Keep only the main points and avoid adding extra information.

        Summary : {context}
        Question : {question}
    """,
    input_variables=['context', 'question']
)

summary_parallel_chain = RunnableParallel({
    'context': summary_retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

summary_chain = summary_parallel_chain | summary_prompt | chat_model | str_parser
class Query_category(BaseModel):
    category : Annotated[
        Literal['summary', 'question_answer'],
        Field(description="The category of the query, either 'summary' or 'question_answer'.")
    ]

pydantic_parser = PydanticOutputParser(pydantic_object = Query_category)
classification_prompt = PromptTemplate(
    template = "Classify the user query into either 'summary' or 'question_answer'.\n User query : {user_query} \n {format_instructions}",
    input_variables = ['user_query'],
    partial_variables = {'format_instructions' : pydantic_parser.get_format_instructions()}
)
query_classification_chain = classification_prompt | chat_model | pydantic_parser

query_extractor = RunnableLambda(lambda x: x['user_query'])

parallel_chain = RunnableParallel({
    'classification': lambda x: query_classification_chain.invoke({'user_query': x}),
    'query': RunnablePassthrough()
})

branch_chain = RunnableBranch(
    (lambda x: x['classification'].category == 'summary', 
    RunnableLambda(lambda x: summary_chain.invoke(x['query']))),
    RunnableLambda(lambda x: qa_chain.invoke(x['query']))
)

final_chain = query_extractor | parallel_chain | branch_chain

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

    result = final_chain.invoke({'user_query': question})
    print(f'\n{result}')