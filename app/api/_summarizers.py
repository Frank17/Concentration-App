from abc import ABC, abstractclassmethod
from math import ceil
from heapq import nlargest
from ._algo import get_sents, cos_summarize, freq_summarize


def select(
    scored_sents: dict[str, float],
    percent: int | float = 0
) -> list[str]:
    n = ceil(percent * len(scored_sents))
    return nlargest(n, scored_sents, key=scored_sents.get)


class BaseSummarizer(ABC):
    @abstractclassmethod
    def _summarize(self, text):
        pass

    def summarize(self, text, percent=0):
        sents = get_sents(text)
        return '. '.join(select(self._summarize(sents), percent)) + '.'


    def categorize(
        self,
        text: str,
        keywords: set[str],
        percent: int | float,
        threhold: int | float
    ) -> bool:
        sents = get_sents(text)
        summary = select(self._summarize(sents), percent)
        words = {word for sent in summary for word in sent.split()}
        jaccard_simi = len(words & keywords) / len(words | keywords)
        return jaccard_simi >= threhold


class CosineSummarizer(BaseSummarizer):
    def _summarize(self, sents):
        return cos_summarize(sents)


class FrequencySummarizer(BaseSummarizer):
    def _summarize(self, text):
        return freq_summarize(text)


# Demo video: https://www.youtube.com/watch?v=-OQ3l6lpzoc