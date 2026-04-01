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


@tool
def query_database(question: str) -> str:
    result = _get_sql_agent()(question)
    return str(result)


@tool
def search_knowledge_base(question: str) -> str:
    result = _get_rag_agent()(question)
    return str(result)


SYSTEM_PROMPT = """You are an assistant orchestrating a multi-agent system.

You have two tools:
- query_database: for structured data (users, transactions, products, sales, orders)
- search_knowledge_base: for unstructured knowledge (policies, documentation, guides)

Analyze the question, call the appropriate tool(s), then write a clear and concise answer.

Formatting rules (the response is shown in Telegram):
- Use HTML tags: <b>text</b> for bold, <code>value</code> for numbers/IDs/names.
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


async def stream(query: str, chat_id: str):
    """Async generator that yields text tokens as they arrive from the orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = create_orchestrator()

    async for event in _orchestrator.stream_async(_with_context(query, chat_id)):
        if "data" in event:
            yield event["data"]


def run(query: str, chat_id: str) -> dict:
    global _orchestrator
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

    sources = _extract_sources(result)

    return {
        "response": response_text,
        "sources": sources,
        "token_usage": token_usage,
    }


def _extract_sources(result) -> list[str]:
    sources = []
    try:
        for msg in result.messages:
            if msg.get("role") == "assistant":
                for block in msg.get("content", []):
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "")
                        if tool_name == "query_database":
                            sources.append("SQL Database")
                        elif tool_name == "search_knowledge_base":
                            sources.append("Knowledge Base (Qdrant)")
    except (AttributeError, TypeError):
        pass
    return list(dict.fromkeys(sources))  
