import boto3
import logging
import time
import uuid
from datetime import datetime, timezone
from app.config import settings

log = logging.getLogger(__name__)


def get_client():
    return boto3.client(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def ensure_table():
    client = get_client()
    try:
        client.create_table(
            TableName=settings.dynamodb_table,
            KeySchema=[
                {"AttributeName": "chat_id", "KeyType": "HASH"},
                {"AttributeName": "message_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "chat_id", "AttributeType": "S"},
                {"AttributeName": "message_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Table takes a moment to become ACTIVE — TTL update fails if called too early
        waiter = client.get_waiter("table_exists")
        waiter.wait(TableName=settings.dynamodb_table)

        client.update_time_to_live(
            TableName=settings.dynamodb_table,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )
        log.info(f"DynamoDB table '{settings.dynamodb_table}' created.")
    except client.exceptions.ResourceInUseException:
        log.info("DynamoDB table already exists.")
    except Exception as e:
        log.warning(f"Could not create DynamoDB table: {e}")


def save_log(
    chat_id: str,
    query: str,
    answer: str,
    usage: dict,
    sources: list[str],
    time_taken: float,
    full_name: str = "",
):
    client = get_client()
    timestamp = datetime.now(timezone.utc).isoformat()
    message_id = f"{str(uuid.uuid4())}-{timestamp}"
    ttl = int(time.time()) + settings.dynamodb_ttl_days * 86400

    item = {
        "chat_id": {"S": chat_id},
        "message_id": {"S": message_id},
        "full_name": {"S": full_name},
        "query": {"S": query},
        "answer": {"S": answer},
        "input_tokens": {"N": str(usage.get("prompt_tokens", 0))},
        "output_tokens": {"N": str(usage.get("completion_tokens", 0))},
        "sources_links": {"L": [{"S": s} for s in sources]},
        "time_taken": {"N": str(round(time_taken, 2))},
        "ttl": {"N": str(ttl)},
    }

    try:
        client.put_item(TableName=settings.dynamodb_table, Item=item)
        log.info(f"Saved log for chat_id={chat_id}")
    except Exception as e:
        log.warning(f"Failed to save log to DynamoDB: {e}")


def get_summary(chat_id: str) -> str:
    """Get the stored rolling summary for this chat session."""
    client = get_client()
    try:
        resp = client.get_item(
            TableName=settings.dynamodb_table,
            Key={
                "chat_id": {"S": chat_id},
                "message_id": {"S": "!summary"},
            },
        )
        return resp.get("Item", {}).get("summary", {}).get("S", "")
    except Exception as e:
        log.warning(f"Failed to get summary: {e}")
        return ""


def update_summary(chat_id: str, summary: str):
    """Persist the rolling summary. No TTL — summaries outlive individual messages."""
    client = get_client()
    try:
        client.put_item(
            TableName=settings.dynamodb_table,
            Item={
                "chat_id": {"S": chat_id},
                "message_id": {"S": "!summary"},
                "summary": {"S": summary},
            },
        )
    except Exception as e:
        log.warning(f"Failed to update summary: {e}")


def get_history(chat_id: str, limit: int = 10) -> list[dict]:
    client = get_client()
    try:
        resp = client.query(
            TableName=settings.dynamodb_table,
            KeyConditionExpression="chat_id = :cid",
            ExpressionAttributeValues={":cid": {"S": chat_id}},
            ScanIndexForward=False,
            Limit=limit,
        )
        return [
            {
                "query": item["query"]["S"],
                "answer": item["answer"]["S"],
            }
            for item in resp.get("Items", [])
        ]
    except Exception as e:
        log.warning(f"Failed to get history from DynamoDB: {e}")
        return []
