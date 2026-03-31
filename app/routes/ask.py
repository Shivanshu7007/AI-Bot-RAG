from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.services.embedding import get_embedding
from app.services.qdrant import search
from app.services.llm import generate_answer
from app.core.config import settings

router = APIRouter()


class AskRequest(BaseModel):
    product_id: int
    question: str


@router.post("/ask")
def ask(request: AskRequest):

    try:
        question = request.question.strip()
        question_lower = question.lower()

        # -----------------------------
        # 🔹 Greeting Handler
        # -----------------------------
        greetings = [
            "hi", "hello", "hey", "hii",
            "namaste", "hello ji", "hi ji"
        ]

        if question_lower in greetings:
            return PlainTextResponse(
                "Hi 👋 I’m your product assistant.\n"
                "You can ask me anything related to this product!"
            )

        collection = f"product_{request.product_id}"

        # -----------------------------
        # 🔹 Generate embedding
        # -----------------------------
        try:
            query_vector = get_embedding(question)
        except Exception:
            return PlainTextResponse(
                "⚠️ I’m having trouble processing your question. Please try again."
            )

        # -----------------------------
        # 🔹 Search Qdrant
        # -----------------------------
        try:
            res = search(collection, query_vector)
        except Exception:
            return PlainTextResponse(
                "⚠️ I’m temporarily unable to access product data. Please try again shortly."
            )

        try:
            results = res.json().get("result", [])
        except (ValueError, AttributeError):
            results = []

        # -----------------------------
        # 🔹 Similarity Filter
        # -----------------------------
        filtered = [
            r for r in results
            if r.get("score", 0) >= settings.SIMILARITY_THRESHOLD
        ]

        # -----------------------------
        # 🔹 Off-topic Friendly Reply
        # -----------------------------
        if not filtered:
            return PlainTextResponse(
                "I’m here to help only with questions related to this product 😊\n"
                "Please ask something about the product and I’ll be happy to help!"
            )

        # -----------------------------
        # 🔹 Build Context Safely
        # -----------------------------
        context_parts = []
        for r in filtered:
            payload = r.get("payload", {})
            if "text" in payload:
                context_parts.append(payload["text"])

        context = "\n".join(context_parts)
        context = context[:settings.MAX_CONTEXT_LENGTH]

        # -----------------------------
        # 🔹 Generate Answer
        # -----------------------------
        try:
            answer = generate_answer(
                context=context,
                question=question
            )
            return PlainTextResponse(answer)

        except Exception:
            return PlainTextResponse(
                "⚠️ I’m having a temporary issue generating the answer. Please try again."
            )

    except Exception:
        return PlainTextResponse(
            "Something went wrong 😔 Please try again."
        )
