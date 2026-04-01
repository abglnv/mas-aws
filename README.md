## Quickstart

**1. Copy and fill in credentials:**

```bash
cp .env.example .env
# Edit .env with your AWS credentials and Telegram bot token
```

**2. Start all services:**

```bash
docker-compose up --build
```

On first startup the app will automatically:

- Create PostgreSQL tables and insert sample data (users, products, transactions)
- Create the Qdrant collection and index sample knowledge base documents
- Create the DynamoDB table with TTL enabled

## API

### `POST /invoke`

```json
// Request
{ "query": "How many completed transactions are there?", "chat_id": "optional-uuid" }

// Response
{
  "response": "There are 7 completed transactions.",
  "sources": ["SQL Database"],
  "token_usage": { "prompt_tokens": 450, "completion_tokens": 80, "total_tokens": 530 }
}
```

## Example Questions

**SQL (structured data):**

- "How many users do we have?"
- "What are the top selling products?"
- "Show me all pending transactions"

**RAG (knowledge base):**

- "What is the return policy?"
- "How long does shipping take?"
- "What payment methods are accepted?"
