from embed_prompt import keyword_prompt
from openai import OpenAI

GROQ_API_KEY=""
MODEL_NAME = "llama-3.1-8b-instant"

llm = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

def generate_keywords_from_chunk(chunk: str) -> list[str]:
    """
    Calls Groq LLM to extract keywords from a policy chunk.
    Returns a list of strings.
    """
    # Render the final prompt
    prompt = keyword_prompt.format(chunk=chunk[:2000])  # truncate to save tokens

    try:
        response = llm.responses.create(
            input=prompt,
            model=MODEL_NAME,
        )
        # Extract plain text
        keywords_text = response.output_text.strip()
        # Split and normalize
        keywords = [kw.strip().lower() for kw in keywords_text.split(",") if kw.strip()]
        return keywords
    except Exception as e:
        print(f"⚠️ LLM keyword generation failed: {e}")
        return []
    