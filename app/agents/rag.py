import logging
from strands import Agent, tool
from strands.models.openai import OpenAIModel

# from strands.models.bedrock import BedrockModel 
# import boto3

from app.config import settings
from app.services.qdrant import hybrid_search

log = logging.getLogger(__name__)


def _make_model():
    return OpenAIModel(model_id=settings.openai_model_id)

    # session = boto3.Session(
    #     aws_access_key_id=settings.aws_access_key_id,
    #     aws_secret_access_key=settings.aws_secret_access_key,
    #     region_name=settings.aws_region,
    # )
    # return BedrockModel(model_id=settings.bedrock_model_id, boto3_session=session)




@tool
def search_docs(query: str) -> str:
    results = hybrid_search(query)
    if not results:
        return "No relevant documents found."

    parts = []
    for r in results:
        parts.append(f"[{r.get('title', 'Document')}] (source: {r.get('source', '')})\n{r.get('text', '')}")
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """You are a RAG agent. Answer questions using the knowledge base.

Steps:
1. Use search_docs to retrieve relevant documents.
2. Answer based only on the retrieved content — be concise.
3. Mention the source at the end (e.g. "Source: kb/return-policy").

Formatting (response shown in Telegram, use HTML):
- <b>text</b> for key terms, <code>value</code> for specific values like dates or prices.
- Use plain hyphens for bullet points."""


def create_agent() -> Agent:
    return Agent(
        model=_make_model(),
        tools=[search_docs],
        system_prompt=SYSTEM_PROMPT,
    )
