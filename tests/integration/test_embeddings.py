from app.config.settings import Settings
from app.embeddings.factory import create_embedding_provider


def test_bge_m3_embedding_provider() -> None:
    settings = Settings(
        embedding_provider="sentence_transformer",
        embedding_model="BAAI/bge-m3",
    )

    provider = create_embedding_provider(settings)

    documents = provider.embed_documents(
        [
            "I prefer Python.",
            "I am building an AI agent.",
        ]
    )

    query = provider.embed_query(
        "What programming language do I prefer?"
    )

    assert len(documents) == 2
    assert len(documents[0]) == 1024
    assert len(documents[1]) == 1024
    assert len(query) == 1024