from sentence_transformers import SentenceTransformer

from app.contracts.errors import EmbeddingError


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model: {model_name}"
            ) from exc

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            raise EmbeddingError(
                "Failed to create document embeddings"
            ) from exc

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text:
            return []

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        except Exception as exc:
            raise EmbeddingError(
                "Failed to create query embedding"
            ) from exc