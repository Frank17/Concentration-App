from __future__ import annotations

from math import ceil
from heapq import nlargest
from ._algo import cos_summarize


PERCENT = 0.4  # Extract 40% of the most relevant sentences from the summary
THRESHOLD = 0.0015  # Min Jaccard index needed to block the website


def select(scored_sents: dict[str, float], percent: int | float = 0) -> list[str]:
    """Return percent% of the most relevant sentences from scored_sents
    """
    n = ceil(percent * len(scored_sents))
    return nlargest(n, scored_sents, key=scored_sents.get)


def get_strict_judgment(text: str, keywords: list[str]) -> bool:
    """Return NLP-based judgment on whether the keywords relate to text & url
    """
    keywords = set(keywords)
    summary = select(cos_summarize(text), PERCENT)
    # Calculate the Jaccard index between the text and the keywords
    words = {word for sent in summary for word in sent.split()}
    jaccard_idx = len(words & keywords) / len(words | keywords)
    if jaccard_idx >= THRESHOLD:
        return 'yes'
    return 'no'