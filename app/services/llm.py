import logging
from openai import OpenAI, OpenAIError
from app.core.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL
)

SYSTEM_PROMPT = """
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


def generate_answer(context: str, question: str, mode: str = "normal") -> str:
    try:
        completion = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],
            temperature=0.2,
            max_tokens=600
        )

        answer = completion.choices[0].message.content.strip()
        logger.info(
            f"[LLM] Generated answer — model={settings.MODEL_NAME} "
            f"prompt_tokens={completion.usage.prompt_tokens} "
            f"completion_tokens={completion.usage.completion_tokens}"
        )
        return answer

    except OpenAIError as e:
        logger.error(f"[LLM] OpenRouter API error: {e}")
        raise
    except Exception as e:
        logger.error(f"[LLM] Unexpected error: {e}")
        raise
