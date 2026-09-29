from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    openai_deployment: str | None = None
    openai_api_version: str | None = None
    openai_endpoint: str | None = None

    llm_provider: str = "azure_openai"
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_provider: str = "chroma"
    vector_store_path: str = "vectorstore/chroma"

    stm_top_k: int = Field(default=5, ge=1)
    ltm_top_k: int = Field(default=5, ge=1)
    pdf_top_k: int = Field(default=5, ge=1)
    context_token_budget: int = Field(default=4000, ge=1)
    max_retries: int = Field(default=2, ge=0)

    conversation_data_path: str = "data/conversations"
    trace_data_path: str = "data/traces"


settings = Settings()
