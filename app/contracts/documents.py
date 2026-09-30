from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    page_number: int | None = None
    section: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class Document(BaseModel):
    document_id: str
    filename: str
    content: str
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content: str
    page_number: int | None = None
    section: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)