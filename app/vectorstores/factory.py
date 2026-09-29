from app.contracts.errors import ConfigurationError


def create_vector_store(provider: str, **kwargs):
    if provider == "chroma":
        from app.vectorstores.chroma import ChromaVectorStore

        return ChromaVectorStore(**kwargs)

    raise ConfigurationError(f"Unsupported vector store provider: {provider}")
