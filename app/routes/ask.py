import re
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.services.embedding import get_embedding
from app.services.qdrant import search
from app.services.llm import generate_answer
from app.core.config import settings
from app.core.security import verify_api_key

router = APIRouter()


class AskRequest(BaseModel):
    product_id: int
    question: str


@router.post("/ask")
def ask(
    request: AskRequest,
    _: str = Depends(verify_api_key)
):

    collection = f"product_{request.product_id}"

    query_vector = get_embedding(request.question)
    res = search(collection, query_vector)

    if res.status_code != 200:
        raise HTTPException(status_code=404, detail="Product not found")

    results = res.json().get("result", [])

    filtered = [
        r for r in results
        if r.get("score", 0) >= settings.SIMILARITY_THRESHOLD
    ]

    if not filtered:
        return PlainTextResponse(
            "I couldn't find relevant information in this product's documents."
        )

    context = "\n".join(
        r["payload"]["text"]
        for r in filtered
        if "payload" in r
    )

    context = context[:settings.MAX_CONTEXT_LENGTH]

    question_lower = request.question.lower()

    general_keywords = [
        "about", "overview", "summary", "introduction",
        "what is this", "what is this product",
        "tell me about", "details about",
        "ye kya hai", "kis kaam ka hai",
        "iske bare me", "batao"
    ]

    if any(keyword in question_lower for keyword in general_keywords):
        mode = "summary"
    else:
        mode = "normal"

    answer = generate_answer(
        context=context,
        question=request.question,
        mode=mode
    )

    return PlainTextResponse(answer)
