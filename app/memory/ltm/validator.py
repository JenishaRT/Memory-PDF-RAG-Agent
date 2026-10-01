from app.contracts.memory import CandidateMemory


class LTMValidator:
    def __init__(
        self,
        *,
        max_content_length: int = 2000,
    ) -> None:
        if max_content_length <= 0:
            raise ValueError(
                "max_content_length must be greater than zero"
            )

        self.max_content_length = max_content_length

    def validate(
        self,
        candidate: CandidateMemory,
    ) -> None:
        if not candidate.user_id.strip():
            raise ValueError(
                "Memory candidate must have a user_id"
            )

        if not candidate.content.strip():
            raise ValueError(
                "Memory candidate content cannot be empty"
            )

        if len(candidate.content) > self.max_content_length:
            raise ValueError(
                "Memory candidate content is too long"
            )

        if not candidate.source.thread_id.strip():
            raise ValueError(
                "Memory candidate must have a source thread_id"
            )

        if not candidate.source.message_ids:
            raise ValueError(
                "Memory candidate must have source message_ids"
            )

        if any(
            not message_id.strip()
            for message_id in candidate.source.message_ids
        ):
            raise ValueError(
                "Memory candidate contains an empty message_id"
            )