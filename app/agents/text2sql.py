import logging
from strands import Agent, tool
from strands.models.openai import OpenAIModel

# from strands.models.bedrock import BedrockModel  
# import boto3

from app.config import settings
from app.services.postgres import execute_query, get_schema

log = logging.getLogger(__name__)


def _make_model():
    return OpenAIModel(model_id=settings.openai_model_id)

    # session = boto3.Session(
    #     aws_access_key_id=settings.aws_access_key_id,
    #     aws_secret_access_key=settings.aws_secret_access_key,
    #     region_name=settings.aws_region,
    # )
    # return BedrockModel(model_id=settings.bedrock_model_id, boto3_session=session)


def _format_rows(rows: list[dict]) -> str:
    """Format query results as a clean ASCII table."""
    headers = list(rows[0].keys())
    col_widths = [
        max(len(h), max(len(str(row.get(h, ""))) for row in rows))
        for h in headers
    ]
    sep = "-+-".join("-" * w for w in col_widths)
    header = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    data = [
        " | ".join(str(row.get(h, "")).ljust(w) for h, w in zip(headers, col_widths))
        for row in rows
    ]
    return "\n".join([header, sep] + data)


_last_sql: list[str] = []


@tool
def run_sql(sql: str) -> str:
    """Execute a SQL SELECT query and return formatted results."""
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."
    log.info(f"run_sql called: {sql.strip()!r}")
    _last_sql.append(sql.strip())
    try:
        rows = execute_query(sql)
        if not rows:
            return "Query returned no results."
        rows = rows[:20]
        return f"{len(rows)} row(s):\n{_format_rows(rows)}"
    except Exception as e:
        return f"SQL Error: {e}"


SYSTEM_PROMPT = f"""You are a Text2SQL agent. Convert questions to SQL and execute them.

{get_schema()}

Rules:
- Only write SELECT queries.
- Use run_sql to execute your query.
- Present the answer as a short, readable summary — highlight the key numbers or names.
- If there are many rows, summarize instead of listing everything.
- Use HTML tags for formatting: <b>label</b> for emphasis, <code>value</code> for numbers/IDs.
- Never dump raw table data directly to the user."""


def create_agent() -> Agent:
    return Agent(
        model=_make_model(),
        tools=[run_sql],
        system_prompt=SYSTEM_PROMPT,
    )


def run(agent: Agent, question: str) -> dict:
    _last_sql.clear()
    result = agent(question)
    return {
        "answer": str(result),
        "sources": list(_last_sql),
    }
