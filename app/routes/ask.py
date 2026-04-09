import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from app.services.embedding import get_embedding
from app.services.qdrant import search, collection_has_documents
from app.services.llm import generate_answer
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_QUESTION_LENGTH = 1000

GREETINGS = {
    "hi", "hello", "hey", "hii", "helo",
    "namaste", "hello ji", "hi ji", "hey there"
}


class HistoryMessage(BaseModel):
    sender: str   # "user" or "bot"
    text: str


class AskRequest(BaseModel):
    product_id: int
    question: str
    history: Optional[List[HistoryMessage]] = None


@router.post("/ask")
def ask(request: AskRequest):

    try:
        question = request.question.strip()

        # -----------------------------
        # Input length guard
        # -----------------------------
        if len(question) > MAX_QUESTION_LENGTH:
            raise HTTPException(status_code=400, detail="Question too long (max 1000 characters)")

        question_lower = question.lower()

        # -----------------------------
        # Greeting Handler
        # -----------------------------
        if question_lower in GREETINGS:
            return PlainTextResponse(
                "How can I help you with this product today?"
            )

        collection = f"product_{request.product_id}"

        # -----------------------------
        # Knowledge base existence check
        # -----------------------------
        try:
            has_docs = collection_has_documents(collection)
        except Exception:
            logger.exception(
                "[Ask] Qdrant collection_has_documents failed for collection=%s", collection
            )
            return PlainTextResponse(
                "I'm temporarily unable to access product data. Please try again shortly."
            )

        if not has_docs:
            return JSONResponse({"status": "no_knowledge_base"})

        # -----------------------------
        # Generate embedding
        # -----------------------------
        try:
            query_vector = get_embedding(question)
        except Exception:
            logger.exception("[Ask] Embedding generation failed for question=%r", question)
            return PlainTextResponse(
                "I'm having trouble processing your question. Please try again."
            )

        # -----------------------------
        # Search Qdrant
        # -----------------------------
        try:
            res = search(collection, query_vector)
        except Exception:
            logger.exception("[Ask] Qdrant search failed for collection=%s", collection)
            return PlainTextResponse(
                "I'm temporarily unable to access product data. Please try again shortly."
            )

        try:
            results = res.json().get("result", [])
        except (ValueError, AttributeError):
            results = []

        # -----------------------------
        # Similarity Filter
        # -----------------------------
        filtered = [
            r for r in results
            if r.get("score", 0) >= settings.SIMILARITY_THRESHOLD
        ]

        # -----------------------------
        # Off-topic Friendly Reply
        # -----------------------------
        if not filtered:
            return PlainTextResponse(
                "I was not able to find this in the product documentation. "
                "Please contact Cellogen support at contact@cellogenbiotech.com "
                "or +91-9217371321 for assistance."
            )

        # -----------------------------
        # Build Context Safely
        # -----------------------------
        context_parts = []
        for r in filtered:
            payload = r.get("payload", {})
            if "text" in payload:
                context_parts.append(payload["text"])

        context = "\n".join(context_parts)

        # Truncate at a word boundary to avoid splitting mid-word
        if len(context) > settings.MAX_CONTEXT_LENGTH:
            context = context[:settings.MAX_CONTEXT_LENGTH].rsplit(" ", 1)[0]

        # -----------------------------
        # Generate Answer
        # -----------------------------
        try:
            answer = generate_answer(
                context=context,
                question=question,
                history=request.history or []
            )
            return PlainTextResponse(answer)

        except Exception:
            logger.exception("[Ask] LLM answer generation failed")
            return PlainTextResponse(
                "I'm having a temporary issue generating the answer. Please try again."
            )

    except HTTPException:
        raise
    except Exception:
        logger.exception("[Ask] Unexpected error in /ask endpoint")
        return PlainTextResponse(
            "Something went wrong. Please try again."
        )
