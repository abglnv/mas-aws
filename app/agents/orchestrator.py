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


def _extract_sql_from_result(result) -> str | None:
    try:
        for msg in result.messages:
            if msg.get("role") == "assistant":
                for block in msg.get("content", []):
                    if block.get("type") == "tool_use" and block.get("name") == "run_sql":
                        sql = block.get("input", {}).get("sql", "").strip()
                        if sql:
                            return sql
    except (AttributeError, TypeError):
        pass
    return None


def _extract_kb_sources_from_result(result) -> list[str]:
    sources = []
    try:
        for msg in result.messages:
            if msg.get("role") == "user":
                for block in msg.get("content", []):
                    if block.get("type") == "tool_result":
                        content = block.get("content", "")
                        if isinstance(content, str):
                            for part in content.split("---"):
                                for line in part.splitlines():
                                    if line.strip().startswith("(source:"):
                                        src = line.strip().removeprefix("(source:").removesuffix(")").strip()
                                        if src:
                                            sources.append(src)
    except (AttributeError, TypeError):
        pass
    return list(dict.fromkeys(sources))


@tool
def query_database(question: str) -> str:
    agent = _get_sql_agent()
    result = agent(question)
    sql = _extract_sql_from_result(result)
    answer = str(result)
    if sql:
        return f"{answer}\n\n[SQL_USED: {sql}]"
    return answer


@tool
def search_knowledge_base(question: str) -> str:
    agent = _get_rag_agent()
    result = agent(question)
    sources = _extract_kb_sources_from_result(result)
    answer = str(result)
    if sources:
        return f"{answer}\n\n[KB_SOURCES: {', '.join(sources)}]"
    return answer


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


async def stream(query: str, chat_id: str):
    """Async generator that yields dicts: {"token": str} or {"status": str}."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = create_orchestrator()

    async for event in _orchestrator.stream_async(_with_context(query, chat_id)):
        if "current_tool_use" in event:
            tool_name = (event["current_tool_use"] or {}).get("name", "")
            status = _TOOL_STATUS.get(tool_name, f"Using {tool_name}...")
            yield {"status": status}
        elif "data" in event:
            yield {"token": event["data"]}


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
            if msg.get("role") == "user":
                for block in msg.get("content", []):
                    if block.get("type") != "tool_result":
                        continue
                    content = block.get("content", "")
                    if not isinstance(content, str):
                        continue
                    if "[SQL_USED:" in content:
                        sql = content.split("[SQL_USED:")[1].split("]")[0].strip()
                        sources.append(f"SQL: {sql}")
                    if "[KB_SOURCES:" in content:
                        kb = content.split("[KB_SOURCES:")[1].split("]")[0].strip()
                        for s in kb.split(", "):
                            if s:
                                sources.append(f"KB: {s}")
    except (AttributeError, TypeError):
        pass
    return list(dict.fromkeys(sources))  
