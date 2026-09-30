from app.config.settings import Settings
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddings


def create_embedding_provider(
    settings: Settings,
) -> SentenceTransformerEmbeddings:
    if settings.embedding_provider != "sentence_transformer":
        raise ValueError(
            f"Unsupported embedding provider: "
            f"{settings.embedding_provider}"
        )

    return SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
    )