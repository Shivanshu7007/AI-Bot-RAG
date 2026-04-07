from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from app.services.embedding import get_embedding
from app.services.qdrant import search, collection_has_documents
from app.services.llm import generate_answer
from app.core.config import settings

router = APIRouter()


from typing import List, Optional

class HistoryMessage(BaseModel):
    sender: str   # "user" or "bot"
    text: str

class AskRequest(BaseModel):
    product_id: int
    question: str
    history: Optional[List[HistoryMessage]] = []


@router.post("/ask")
def ask(request: AskRequest):

    try:
        question = request.question.strip()
        question_lower = question.lower()

        # -----------------------------
        # 🔹 Greeting Handler
        # -----------------------------
        greetings = {
            "hi", "hello", "hey", "hii", "helo",
            "namaste", "hello ji", "hi ji", "hey there"
        }

        if question_lower in greetings:
            return PlainTextResponse(
                "How can I help you with this product today? 😊"
            )

        collection = f"product_{request.product_id}"

        # -----------------------------
        # 🔹 Knowledge base existence check
        # -----------------------------
        try:
            has_docs = collection_has_documents(collection)
        except Exception:
            return PlainTextResponse(
                "⚠️ I'm temporarily unable to access product data. Please try again shortly."
            )

        if not has_docs:
            return JSONResponse({"status": "no_knowledge_base"})

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
                "I was not able to find this in the product documentation. "
                "Please contact Cellogen support at contact@cellogenbiotech.com "
                "or +91-9217371321 for assistance."
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
                question=question,
                history=request.history or []
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
