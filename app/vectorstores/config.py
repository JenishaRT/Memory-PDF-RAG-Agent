from pydantic import BaseModel


class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    persist_directory: str = "vectorstore/chroma"
