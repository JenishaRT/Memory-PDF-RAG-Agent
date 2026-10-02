import re

# Words that carry no meaning when comparing a question to a memory
STOP_WORDS = {
    "a", "an", "and", "are", "about", "as", "at", "be", "by", "can", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "please", "that", "the", "this", "to", "was", "what",
    "whats", "which", "with", "you", "your", "tell", "much", "many",
}


def keywords(text):
    """Turns text into a set of meaningful lowercase words.

    'What is the HP of the cooling tower fan?' -> {'hp', 'cooling', 'tower', 'fan'}
    """
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in STOP_WORDS}


def normalize_space(text):
    """Lowercases text and collapses all whitespace, so line breaks don't matter"""
    return " ".join((text or "").lower().split())
