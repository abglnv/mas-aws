import logging
import boto3
import watchtower
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app.routers import api
from app.db import init_all


def setup_logging():
    log_handlers = [logging.StreamHandler()]  

    # Add CloudWatch handler
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        try:
            cw_client = boto3.client(
                "logs",
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region,
            )
            log_handlers.append(
                watchtower.CloudWatchLogHandler(
                    log_group_name="/mas-aws/api",
                    boto3_client=cw_client,
                )
            )
        except Exception as e:
            print(f"CloudWatch logging unavailable: {e}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=log_handlers,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_all()
    yield


app = FastAPI(title="MAS RAG System", lifespan=lifespan)
app.include_router(api.router)
