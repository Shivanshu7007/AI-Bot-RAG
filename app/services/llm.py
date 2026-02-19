from openai import OpenAI
from app.core.config import settings

client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL
)


def generate_answer(context: str, question: str, mode: str = "normal"):

    system_prompt = """
You are a friendly and professional product assistant.

IMPORTANT RULES:
- Answer ONLY using the provided context.
- Do NOT invent or guess information.
- If the question is not related to the provided context,
  politely explain that you only answer product-specific questions.
- Respond in the same language style as the user (English or Hinglish).
- Keep the tone friendly and helpful.
- Do NOT use markdown symbols like ** or ###.
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
        max_tokens=600
    )

    return completion.choices[0].message.content.strip()
