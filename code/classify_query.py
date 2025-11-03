from typing import Literal
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from pydantic.v1 import BaseModel, Field
from base_llm import llm


class QueryClassification(BaseModel):
    """The classification of the user's query."""
    classification: Literal["POLICY", "RETURN", "PRODUCT_SEARCH", "INVALID"] = Field(
        ...,
        description=(
            "The category of the query. "
            "POLICY for policies like store timings, return policies, offers, and more. "
            "RETURN for returns of bought products. "
            "PRODUCT_SEARCH for products, recommendations. "
            "INVALID for off-topic, greetings."
        )
    )


structured_llm = llm.with_structured_output(QueryClassification)

system_prompt = """You are a classifier for a fashion store chatbot. Your job is to categorize the user's query into one of four categories:
    1.  POLICY: For any questions about shipping, refunds, store policies, order status, or account issues.
    2.  RETURN: For any questions about returns of bought products.
    3.  PRODUCT_SEARCH: For any questions about finding specific items, browsing, product availability, recommendations, or product details.
    4.  INVALID: For queries that are off-topic, nonsensical, greetings, or unrelated to fashion retail.
    """

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

classifier_chain = prompt | structured_llm

def classify_query(query, history):
    return classifier_chain.invoke({"query": query, "chat_history": history}).classification