import logging
from app.services.postgres import init_db
from app.services.qdrant import ensure_collection, seed_knowledge_base
from app.services.dynamodb import ensure_table

log = logging.getLogger(__name__)


def init_all():
    log.info("Initing postgres")
    init_db()

    log.info("Initing qdrant")
    ensure_collection()
    seed_knowledge_base()

    log.info("Initing dynamodb")
    ensure_table()

    log.info("All databases inited.")
