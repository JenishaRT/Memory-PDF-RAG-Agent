from app.config.settings import get_settings
from app.contracts.runtime import AgentRequest
from app.graph.graph import Phase0Graph
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import (
    JsonlConversationStore,
)


def main() -> None:
    settings = get_settings()

    store = JsonlConversationStore(
        settings.conversation_data_path
    )

    runtime = AgentRuntime(
        conversation_store=store,
        graph=Phase0Graph(),
    )

    print("=" * 60)
    print("Conversational Memory Agent")
    print("Phase 0")
    print("=" * 60)

    user_id = input("User ID: ").strip()
    thread_id = input("Thread ID: ").strip()

    while True:
        message = input("\nYou: ").strip()

        if message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not message:
            continue

        response = runtime.handle(
            AgentRequest(
                user_id=user_id,
                thread_id=thread_id,
                message=message,
            )
        )

        print(f"\nAssistant: {response.answer}")
        print(f"Trace ID: {response.trace_id}")


if __name__ == "__main__":
    main()