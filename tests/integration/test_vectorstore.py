from app.vectorstores.chroma import ChromaVectorStore


def test_chroma_add_search_delete(tmp_path) -> None:
    store = ChromaVectorStore(
        str(tmp_path / "chroma")
    )

    store.add(
        collection="stm_collection",
        ids=["message_1"],
        documents=["I prefer Python."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[
            {
                "source": "stm",
                "user_id": "user_001",
                "thread_id": "thread_001",
            }
        ],
    )

    results = store.search(
        collection="stm_collection",
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
        filters={
            "user_id": "user_001",
            "thread_id": "thread_001",
        },
    )

    assert len(results) == 1
    assert results[0].item_id == "message_1"
    assert results[0].content == "I prefer Python."

    store.delete(
        collection="stm_collection",
        ids=["message_1"],
    )

    results = store.search(
        collection="stm_collection",
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
        filters={
            "user_id": "user_001",
            "thread_id": "thread_001",
        },
    )

    assert results == []