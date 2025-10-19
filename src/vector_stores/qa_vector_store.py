from langchain_community.vectorstores import FAISS

def build_qa_vector_store(chunks, embedding_model):
    qa_vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return qa_vector_store