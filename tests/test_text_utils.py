import pytest

from llm_eval.text_utils import (
    bullet_items,
    contains_phrase,
    content_terms,
    count_syllables,
    flesch_reading_ease,
    normalize_term,
    numbered_items,
    split_sentences,
    tokenize,
    unique_terms,
)


def test_tokenize_lowercases_and_keeps_contractions():
    assert tokenize("Earth's axis IS tilted, isn't it?") == [
        "earth's",
        "axis",
        "is",
        "tilted",
        "isn't",
        "it",
    ]


@pytest.mark.parametrize(
    "words",
    [
        ("cause", "causes", "caused", "causing"),
        ("reverse", "reversed", "reverses"),
        ("exercise", "exercises"),
        ("plant", "plants"),
        ("process", "processes"),
    ],
)
def test_normalize_term_merges_simple_inflections(words):
    assert len({normalize_term(w) for w in words}) == 1


def test_normalize_term_does_not_strip_double_s():
    assert normalize_term("glass") == "glass"


def test_content_terms_drop_stopwords_and_optionally_numbers():
    assert content_terms("The cat and the 3 dogs") == ["cat", "3", "dog"]
    assert content_terms("The cat and the 3 dogs", skip_numbers=True) == ["cat", "dog"]


def test_unique_terms_preserves_order():
    assert unique_terms("dogs chase cats; cats chase dogs") == ["dog", "chas", "cat"]


def test_split_sentences_handles_lists_and_punctuation():
    text = "First point. Second point!\n- bullet one\n2) numbered item"
    assert split_sentences(text) == ["First point.", "Second point!", "bullet one", "numbered item"]


def test_split_sentences_empty():
    assert split_sentences("   \n  ") == []


@pytest.mark.parametrize(
    "word, expected", [("cat", 1), ("water", 2), ("table", 2), ("photosynthesis", 5), ("", 0)]
)
def test_count_syllables(word, expected):
    assert count_syllables(word) == expected


def test_flesch_easy_text_scores_higher_than_hard_text():
    easy = "The cat sat on the mat. It was warm."
    hard = (
        "Photosynthetic organisms utilise electromagnetic radiation, transforming "
        "it into biochemically accessible energy via sophisticated mechanisms."
    )
    assert flesch_reading_ease(easy) > flesch_reading_ease(hard)
    assert flesch_reading_ease("") is None


def test_list_detection():
    text = "- a\n* b\n• c\n1. one\n2) two\nplain"
    assert len(bullet_items(text)) == 3
    assert len(numbered_items(text)) == 2


def test_contains_phrase_is_whole_word_and_case_insensitive():
    assert contains_phrase("Let me KNOW if that helps", "let me know")
    assert not contains_phrase("capitalize", "capital")
    assert not contains_phrase("anything", "")
