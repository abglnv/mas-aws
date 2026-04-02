import logging
from strands import Agent, tool
from strands.models.openai import OpenAIModel

# from strands.models.bedrock import BedrockModel 
# import boto3

from app.config import settings
from app.agents import text2sql, rag
from app.services.memory import build_context

log = logging.getLogger(__name__)

# we init the agents as None and give them the value in the function 
# so that agents will be inited only once 

_sql_agent = None
_rag_agent = None


def _get_sql_agent():
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = text2sql.create_agent()
    return _sql_agent


def _get_rag_agent():
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = rag.create_agent()
    return _rag_agent


_request_sources: list[str] = []


@tool
def query_database(question: str) -> str:
    result = text2sql.run(_get_sql_agent(), question)
    log.info(f"query_database sources: {result['sources']}")
    _request_sources.extend(result["sources"])
    return result["answer"]


@tool
def search_knowledge_base(question: str) -> str:
    result = rag.run(_get_rag_agent(), question)
    log.info(f"search_knowledge_base sources: {result['sources']}")
    _request_sources.extend(result["sources"])
    return result["answer"]


SYSTEM_PROMPT = """You are an assistant orchestrating a multi-agent system.

You have two tools:
- query_database: for structured data (users, transactions, products, sales, orders)
- search_knowledge_base: for unstructured knowledge (policies, documentation, guides)

Analyze the question, call the appropriate tool(s), then write a clear and concise answer.

If the user asks what you can do, who you are, or how you work — answer directly from this prompt without calling any tools.

Formatting rules (the response is shown in Telegram using HTML parse mode):
- NEVER use Markdown syntax (**bold**, *italic*, `code`, # headers). It will not render.
- Use ONLY HTML tags: <b>text</b> for bold, <i>text</i> for italic, <code>value</code> for numbers/IDs/names.
- Keep answers short. Summarize data — do not paste raw rows or tables.
- Use bullet points (plain hyphens) for lists."""


def create_orchestrator() -> Agent:
    model = OpenAIModel(model_id=settings.openai_model_id)

    # session = boto3.Session(
    #     aws_access_key_id=settings.aws_access_key_id,
    #     aws_secret_access_key=settings.aws_secret_access_key,
    #     region_name=settings.aws_region,
    # )
    # model = BedrockModel(model_id=settings.bedrock_model_id, boto3_session=session)

    return Agent(
        model=model,
        tools=[query_database, search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
    )


_orchestrator = None


def _with_context(query: str, chat_id: str) -> str:
    context = build_context(chat_id)
    if not context:
        return query
    return f"{context}\n\n[Current question]\n{query}"


_TOOL_STATUS = {
    "query_database": "Querying database...",
    "search_knowledge_base": "Searching knowledge base...",
}


_last_result = None


async def stream(query: str, chat_id: str):
    """Async generator that yields dicts: {"token": str} or {"status": str}."""
    global _orchestrator, _last_result
    _request_sources.clear()
    if _orchestrator is None:
        _orchestrator = create_orchestrator()

    async for event in _orchestrator.stream_async(_with_context(query, chat_id)):
        if "current_tool_use" in event:
            tool_name = (event["current_tool_use"] or {}).get("name", "")
            status = _TOOL_STATUS.get(tool_name, f"Using {tool_name}...")
            yield {"status": status}
        elif "data" in event:
            yield {"token": event["data"]}
        elif "result" in event:
            _last_result = event["result"]


def extract_last_sources(_chat_id: str) -> list[str]:
    return list(dict.fromkeys(_request_sources))


def extract_last_usage() -> dict:
    global _last_result
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if _last_result is None:
        return usage
    try:
        acc = _last_result.metrics.accumulated_usage
        usage["prompt_tokens"] = acc.get("inputTokens", 0)
        usage["completion_tokens"] = acc.get("outputTokens", 0)
    except (AttributeError, KeyError):
        pass
    return usage


def run(query: str, chat_id: str) -> dict:
    global _orchestrator
    _request_sources.clear()
    if _orchestrator is None:
        _orchestrator = create_orchestrator()

    result = _orchestrator(_with_context(query, chat_id))
    response_text = str(result)

    # token usage 
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        usage = result.metrics.accumulated_usage
        token_usage = {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        }
    except (AttributeError, KeyError):
        pass

    return {
        "response": response_text,
        "sources": list(dict.fromkeys(_request_sources)),
        "token_usage": token_usage,
    }
