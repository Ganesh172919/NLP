from __future__ import annotations

import nltk


def main() -> None:
    # Minimal set for common tutorials
    for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg)


if __name__ == "__main__":
    main()
