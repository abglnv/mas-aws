import time
import json
import asyncio
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.schemas import InvokeRequest, InvokeResponse, TokenUsage
from app.agents import orchestrator
from app.services.dynamodb import save_log
from app.services.memory import roll_summary

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest):
    log.info(f"[chat_id={request.chat_id}] query: {request.query!r}")
    start = time.time()

    # asyncio so we don't block the fastapi
    result = await asyncio.get_event_loop().run_in_executor(
        None, orchestrator.run, request.query, request.chat_id
    )

    time_taken = time.time() - start
    log.info(f"[chat_id={request.chat_id}] done in {time_taken:.2f}s | sources={result['sources']}")

    asyncio.get_event_loop().run_in_executor(
        None,
        save_log,
        request.chat_id,
        request.query,
        result["response"],
        result["token_usage"],
        result["sources"],
        time_taken,
    )
    asyncio.get_event_loop().run_in_executor(None, roll_summary, request.chat_id)

    return InvokeResponse(
        response=result["response"],
        sources=result["sources"],
        token_usage=TokenUsage(**result["token_usage"]),
    )


@router.post("/invoke/stream")
async def invoke_stream(request: InvokeRequest):
    log.info(f"[chat_id={request.chat_id}] stream query: {request.query!r}")
    start = time.time()

    async def generate():
        async for event in orchestrator.stream(request.query, request.chat_id):
            yield f"data: {json.dumps(event)}\n\n"
        log.info(f"[chat_id={request.chat_id}] stream done in {time.time() - start:.2f}s")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
