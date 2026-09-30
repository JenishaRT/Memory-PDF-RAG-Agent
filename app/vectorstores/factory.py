from app.config.settings import Settings
from app.vectorstores.chroma import ChromaVectorStore


def create_vector_store(settings: Settings) -> ChromaVectorStore:
    if settings.vector_store_type != "chroma":
        raise ValueError(
            f"Unsupported vector store: {settings.vector_store_type}"
        )

    return ChromaVectorStore(
        path=settings.vector_store_path,
    )