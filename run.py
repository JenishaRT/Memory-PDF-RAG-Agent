from __future__ import annotations

import argparse

from app.main import create_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production LangGraph Agent")
    parser.add_argument("--user", required=True, help="User identifier")
    parser.add_argument("--thread", required=True, help="Conversation thread identifier")
    parser.add_argument("--query", help="Run one query instead of interactive mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = create_runtime()

    if args.query:
        response = runtime.run(args.user, args.thread, args.query)
        print(response.answer)
        print(f"trace_id={response.trace_id}")
        return

    print("Agent started")
    print(f"User: {args.user}")
    print(f"Thread: {args.thread}")
    print("Type 'exit' to quit.")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if message.lower() == "exit":
            break
        if not message:
            continue

        response = runtime.run(args.user, args.thread, message)
        print(f"Agent: {response.answer}")
        print(f"trace_id={response.trace_id}")


if __name__ == "__main__":
    main()
