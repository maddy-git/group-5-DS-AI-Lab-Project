# 1️⃣  Create prompt template (LangChain style)
from langchain_core.prompts import PromptTemplate
keyword_prompt = PromptTemplate(
    input_variables=["chunk"],
    template=(
        "You are an expert retail policy analyst. "
        "From the following text, extract 3–6 concise, meaningful keywords "
        "that best represent its main topics or themes.\n\n"
        "Text:\n{chunk}\n\n"
        "Output only the keywords, comma-separated, no explanations."
    ),
)