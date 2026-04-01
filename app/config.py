from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    openai_model_id: str = "gpt-4o-mini"

    # AWS creds
    # aws_region: str = "us-east-1"
    # aws_access_key_id: str = ""
    # aws_secret_access_key: str = ""

    # Bedrock
    # bedrock_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    # DynamoDB
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # Postgres
    postgres_url: str = "postgresql://mas_user:mas_password@postgres:5432/mas_db"

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "knowledge_base"

    # DynamoDB
    dynamodb_table: str = "mas_conversations"
    dynamodb_ttl_days: int = 30

    # Telegram bot
    telegram_token: str = ""
    api_url: str = "http://api:8000"

    class Config:
        env_file = ".env"


settings = Settings()
