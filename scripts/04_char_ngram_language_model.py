from __future__ import annotations

from collections import Counter, defaultdict


def train_char_ngram(text: str, n: int = 4):
    if n < 2:
        raise ValueError("n must be >= 2")

    counts = defaultdict(Counter)
    padded = "~" * (n - 1) + text

    for i in range(len(padded) - (n - 1)):
        context = padded[i : i + (n - 1)]
        nxt = padded[i + (n - 1)]
        counts[context][nxt] += 1

    return counts


def generate(model, n: int, max_len: int = 200, seed: str = "") -> str:
    context = ("~" * (n - 1) + seed)[-(n - 1) :]
    out = list(seed)

    for _ in range(max_len):
        dist = model.get(context)
        if not dist:
            break
        nxt = dist.most_common(1)[0][0]
        out.append(nxt)
        context = (context + nxt)[-(n - 1) :]

    return "".join(out)


def main() -> None:
    text = (
        "natural language processing (nlp) is fun. "
        "start small, iterate, and learn by coding. "
    )

    n = 5
    model = train_char_ngram(text, n=n)
    print(generate(model, n=n, seed="nlp ", max_len=120))


if __name__ == "__main__":
    main()
