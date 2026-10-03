import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_PATH = os.path.join(os.path.dirname(__file__), "..", "qdrant_db")
COLLECTION_NAME = "my_docs"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 embedding dimension


def get_qdrant_client() -> QdrantClient | None:
    configured_url = os.getenv("QDRANT_URL")
    configured_key = os.getenv("QDRANT_API_KEY")

    if configured_url:
        client = QdrantClient(url=configured_url, api_key=configured_key)
        try:
            client.get_collections()
            return client
        except Exception:
            print("[db] Remote Qdrant unavailable; falling back to local storage")

    try:
        client = QdrantClient(path=QDRANT_PATH)
        client.get_collections()
        return client
    except Exception as exc:
        print(
            "[db] Local Qdrant storage unavailable. "
            f"This usually means another Qdrant client is already using the local database ({exc}). "
            "Configure QDRANT_URL or stop the other instance to enable the local DB."
        )
        return None


# Backward-compatible alias used across the project.
client = get_qdrant_client()
