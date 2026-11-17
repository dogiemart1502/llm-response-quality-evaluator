"""Small, dependency-free text-processing helpers.

These are intentionally simple and deterministic so that every automated score
can be explained and reproduced. They are not linguistically sophisticated.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+\S")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+\S")

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just let me more most my myself no nor not of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    """.split()
)

#: Words that describe *how* to answer rather than *what* the topic is. They are
#: excluded when extracting the topical terms of a prompt.
INSTRUCTION_WORDS = frozenset(
    """
    answer bullet bulleted brief briefly concise describe detail details explain format give
    include json list mention no numbered object paragraph please point points provide response
    return sentence sentences short simple simply summarise summarize summary tell using word
    words write one two three four five six seven eight nine ten than more less most least
    keys key
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def normalize_term(token: str) -> str:
    """Crude normalisation so that simple inflections match.

    Strips a possessive ``'s``, a plural ``s`` (not after ``ss``/``us``/``is``),
    an ``-ing``/``-ed`` ending on longer words and a trailing ``e``, then
    truncates to 7 characters, so "cause", "causes" and "caused" all map to
    ``"caus"``. This is deliberately lightweight; it is a known source of both
    false matches and misses.
    """
    t = token.lower()
    if t.endswith("'s"):
        t = t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 5 and t.endswith("ed"):
        t = t[:-2]
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    return t[:7]


def content_terms(
    text: str, extra_stopwords: frozenset[str] = frozenset(), *, skip_numbers: bool = False
) -> list[str]:
    """Return normalised, non-stopword terms in order of appearance (with repeats).

    Args:
        text: Input text.
        extra_stopwords: Additional (un-normalised, lowercase) words to drop.
        skip_numbers: Drop purely numeric tokens (useful for prompts, where
            numbers are usually length limits rather than topics).
    """
    excluded = STOPWORDS | extra_stopwords
    return [
        normalize_term(t)
        for t in tokenize(text)
        if t not in excluded and not (skip_numbers and t.isdigit())
    ]


def unique_terms(
    text: str, extra_stopwords: frozenset[str] = frozenset(), *, skip_numbers: bool = False
) -> list[str]:
    """Like :func:`content_terms` but de-duplicated, preserving order."""
    return list(dict.fromkeys(content_terms(text, extra_stopwords, skip_numbers=skip_numbers)))


def word_count(text: str) -> int:
    return len(tokenize(text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Line breaks are treated as boundaries so that list items count as separate
    units, and list markers are stripped.
    """
    sentences: list[str] = []
    for line in text.splitlines():
        line = _LIST_MARKER_RE.sub("", line).strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_END_RE.split(line) if s.strip())
    return sentences


def count_syllables(word: str) -> int:
    """Estimate syllables by counting vowel groups (English heuristic)."""
    w = word.lower()
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float | None:
    """Flesch reading ease (higher = easier). ``None`` for text without words."""
    words = tokenize(text)
    if not words:
        return None
    sentences = max(len(split_sentences(text)), 1)
    syllables = sum(count_syllables(w) for w in words if w.isalpha()) or len(words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def bullet_items(text: str) -> list[str]:
    return [line for line in text.splitlines() if _BULLET_RE.match(line)]


def numbered_items(text: str) -> list[str]:
    return [line for line in text.splitlines() if _NUMBERED_RE.match(line)]


def contains_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive whole-word phrase match (whitespace-normalised)."""
    words = tokenize(phrase)
    if not words:
        return False
    pattern = r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None
