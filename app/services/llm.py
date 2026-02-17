from openai import OpenAI
from app.core.config import settings

client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL
)


def generate_answer(context: str, question: str, mode: str = "normal"):

    if mode == "summary":
        system_prompt = """
You are a helpful product assistant.

Rules:
- Answer ONLY from provided context.
- Do NOT invent information.
- If context does not contain the answer, say:
  "I couldn't find relevant information in this product's documents."
- Keep tone friendly and professional.
- Do NOT use markdown symbols like ** or ###.
- Use clean formatting.
"""
    else:
        system_prompt = """
You are a strict retrieval-based assistant.

Rules:
- Answer ONLY from provided context.
- Do NOT invent information.
- If answer is not found, say:
  "I couldn't find relevant information in this product's documents."
- If user asks for bullet points, respond using '-' dash format.
- Do NOT use markdown symbols like **.
- Keep formatting clean and readable.
"""

    completion = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    return completion.choices[0].message.content.strip()
