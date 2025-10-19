from src.prompt_templates.prompt import classification_prompt_template

def build_classification_chain(chat_model, pydantic_parser):
    classification_prompt = classification_prompt_template(pydantic_parser)

    query_classification_chain = classification_prompt | chat_model | pydantic_parser
    
    return query_classification_chain