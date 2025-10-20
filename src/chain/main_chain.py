from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from src.chain.classification_chain import build_classification_chain
from src.schema.query_category import Query_category

def build_final_chain(chat_model, qa_chain, summary_chain):
    pydantic_parser = PydanticOutputParser(pydantic_object = Query_category)
    query_classification_chain = build_classification_chain(chat_model, pydantic_parser)

    parallel_chain = RunnableParallel({
        'classification': lambda user_query: query_classification_chain.invoke(user_query),
        'query': RunnablePassthrough()
    })

    branch_chain = RunnableBranch(
        (lambda x: x['classification'].category == 'summary',
            RunnableLambda(lambda x: summary_chain.invoke(x['query']))),
        RunnableLambda(lambda x: qa_chain.invoke(x['query']))
    )
    
    query_extractor = RunnableLambda(lambda user_query: user_query)
    final_chain = query_extractor | parallel_chain | branch_chain

    return final_chain