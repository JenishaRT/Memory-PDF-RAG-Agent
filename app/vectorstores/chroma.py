class ChromaVectorStore:
    """Phase 0 placeholder for the Chroma adapter."""

    def __init__(self, collection_name: str, persist_directory: str) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def add(self, items):
        raise NotImplementedError

    def search(self, query):
        raise NotImplementedError

    def update(self, items):
        raise NotImplementedError

    def delete(self, ids):
        raise NotImplementedError
