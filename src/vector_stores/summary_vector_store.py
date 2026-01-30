from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

def summarize_chunks(chunks: list[Document], chat_model) -> list[Document]:
    prompt = ChatPromptTemplate.from_template("Summarize the following content concisely:\n\n{context}")
    
    summarize_chain = prompt | chat_model | StrOutputParser()
    
    final_summaries = []
    
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        
        summary_text = summarize_chain.invoke({"context": chunk.page_content})
        
        final_summaries.append(Document(page_content = summary_text))
        
        if i != len(chunks) - 1:
            time.sleep(8)

    print("All chunks summarized successfully!")
    return final_summaries

def build_summary_vector_store(chunks: list[Document], chat_model, embedding_model) -> FAISS:
    final_summaries = summarize_chunks(chunks, chat_model)
    summary_vector_store = FAISS.from_documents(
        documents=final_summaries,
        embedding=embedding_model
    )
    return summary_vector_store