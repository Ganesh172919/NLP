from __future__ import annotations

from nlp_learning.text_cleaning import clean_text
from nlp_learning.tokenization import ngrams, simple_tokenize


def main() -> None:
    raw = "I can't believe this works!!! Visit https://example.com now. Email me: a@b.com"
    cleaned = clean_text(raw)
    tokens = simple_tokenize(cleaned)

    print("RAW:", raw)
    print("CLEANED:", cleaned)
    print("TOKENS:", tokens)
    print("BIGRAMS:", ngrams(tokens, 2)[:10])


if __name__ == "__main__":
    main()
