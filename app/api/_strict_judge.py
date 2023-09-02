from __future__ import annotations

from math import ceil
from heapq import nlargest
from ._algo import cos_summarize


PERCENT = 0.4
THRESHOLD = 0.0015


def select(
    scored_sents: dict[str, float],
    percent: int | float = 0
) -> list[str]:
    n = ceil(percent * len(scored_sents))
    return nlargest(n, scored_sents, key=scored_sents.get)


def get_strict_judgment(text: str, keywords: list[str]) -> bool:
    keywords = set(keywords)
    summary = select(cos_summarize(text), PERCENT)
    words = {word for sent in summary for word in sent.split()}
    jaccard_simi = len(words & keywords) / len(words | keywords)
    if jaccard_simi >= THRESHOLD:
        return 'yes'
    return 'no'