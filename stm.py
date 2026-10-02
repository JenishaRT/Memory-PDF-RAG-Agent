"""Short-Term Memory (STM): what happened earlier in THIS session.

STM lives inside the LangGraph state as the `stm_turns` list. The graph is
compiled with a checkpointer, so every call that uses the same session id
(thread_id) sees the turns from the calls before it. When the program exits,
STM is gone - that is what makes it "short-term".
"""
import re

from memory_utils import keywords

STM_MAX_TURNS = 10      # only the most recent turns are kept
STM_ALWAYS_RECENT = 2   # last N turns are passed along for follow-ups ("it", "that")

FOLLOW_UP_PHRASES = (
    "what about", "how about", "and the", "same for", "previous",
    "last answer", "earlier", "that one", "this one",
)
REFERENCE_WORDS = {
    "it", "its", "this", "that", "these", "those", "they", "them",
    "same", "also", "too",
}


def is_follow_up(query):
    """True when the question only makes sense with earlier turns.

    'What about the pump?' and 'Is that value per cell?' are follow-ups.
    """
    q = query.lower().strip()
    if any(phrase in q for phrase in FOLLOW_UP_PHRASES):
        return True
    words = set(re.findall(r"[a-z']+", q))
    return bool(words & REFERENCE_WORDS) or len(words) <= 3


def turn_text(turn):
    """All searchable text of one stored turn"""
    parts = [
        turn.get("query"), turn.get("resolved_query"), turn.get("answer"),
        turn.get("entity"), turn.get("attribute"), turn.get("value"),
    ]
    return " ".join(p for p in parts if p)


def find_relevant_turns(turns, query):
    """Returns the STM turns that help answer this query.

    - Older turns are used only if they share keywords with the query.
    - The most recent turns are used if the query is a follow-up, or if they
      share keywords with it.
    """
    if not turns:
        return []

    query_words = keywords(query)
    recent = turns[-STM_ALWAYS_RECENT:]
    older = turns[:-STM_ALWAYS_RECENT]

    relevant = [t for t in older if query_words & keywords(turn_text(t))]
    recent_overlap = any(query_words & keywords(turn_text(t)) for t in recent)
    if is_follow_up(query) or recent_overlap:
        relevant += recent
    return relevant


def add_turn(turns, turn):
    """Returns a new list with the turn appended, trimmed to STM_MAX_TURNS.

    Asking the exact same question twice in a row doesn't add a duplicate.
    """
    turns = list(turns or [])
    if turns and turns[-1]["query"].lower() == turn["query"].lower():
        turn["turn"] = turns[-1]["turn"]
        turns[-1] = turn
    else:
        turn["turn"] = turns[-1]["turn"] + 1 if turns else 1
        turns.append(turn)
    return turns[-STM_MAX_TURNS:]


def format_turns(turns):
    """Formats STM turns for the prompt, each labeled [STM-n]"""
    if not turns:
        return "(nothing relevant from this conversation)"
    lines = []
    for t in turns:
        lines.append(
            f"[STM-{t['turn']}] User asked: {t['query']}\n"
            f"    Understood as: {t.get('resolved_query') or t['query']}\n"
            f"    Answer: {t['answer']} (source: {t['source']})"
        )
    return "\n".join(lines)
