import re
import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseVector, Prefetch, FusionQuery, Fusion,
)
from fastembed import TextEmbedding, SparseTextEmbedding
from app.config import settings

log = logging.getLogger(__name__)

client = QdrantClient(url=settings.qdrant_url)
dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")

DENSE_DIM = 384
KB_FILE = Path(__file__).parent.parent.parent / "data" / "knowledge_base.md"


def parse_knowledge_base(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")

    sections = re.split(r"\n(?=## )", text)

    docs = []
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        header_line = lines[0].strip()

        if not header_line.startswith("## "):
            continue

        title = header_line.lstrip("# ").strip()
        body = "\n".join(line for line in lines[1:] if line.strip())

        if not body:
            continue

        source = "kb/" + title.lower().replace(" ", "-")
        docs.append({"title": title, "text": body, "source": source})

    log.info(f"Parsed {len(docs)} sections from {path.name}")
    return docs


def ensure_collection():
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    log.info("Qdrant collection created.")


def seed_knowledge_base():
    count = client.count(settings.qdrant_collection).count
    if count > 0:
        log.info("Knowledge base already seeded, skipping.")
        return

    docs = parse_knowledge_base(KB_FILE)
    if not docs:
        log.warning(f"No documents parsed from {KB_FILE}. Check the file format.")
        return

    texts = [doc["text"] for doc in docs]
    dense_vecs = list(dense_model.embed(texts))
    sparse_vecs = list(sparse_model.embed(texts))

    points = [
        {
            "id": i,
            "vector": {
                "dense": dense.tolist(),
                "sparse": SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            "payload": doc,
        }
        for i, (doc, dense, sparse) in enumerate(zip(docs, dense_vecs, sparse_vecs))
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    log.info(f"Seeded {len(points)} documents into knowledge base.")


def hybrid_search(query: str, limit: int = 4) -> list[dict]:
    dense_vec = list(dense_model.embed([query]))[0].tolist()
    sparse_vec = list(sparse_model.embed([query]))[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=limit * 2),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=limit * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    return [
        {"score": p.score, **p.payload}
        for p in results.points
    ]
