from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
import time

def summarize_chunks(chunks, chat_model):
    chunk_summary_chain = load_summarize_chain(chat_model, chain_type='stuff')
    final_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}:")
        summary_output = chunk_summary_chain.invoke([chunk])
        summary_text = summary_output['output_text']
        final_summaries.append(Document(page_content=summary_text))
        if(i != len(chunks) - 1):
            time.sleep(8)

    print("All chunks summarized successfully!")
    return final_summaries

def build_summary_vector_store(chunks, chat_model, embedding_model):
    final_summaries = summarize_chunks(chunks, chat_model)
    summary_vector_store = FAISS.from_documents(
        documents = final_summaries,
        embedding = embedding_model
    )

    return summary_vector_store