from abc import ABC, abstractclassmethod
from ._algo import select, get_sents, cosine_summarize, freq_summarize
from functools import partial


class BaseSummarizer(ABC):
    @abstractclassmethod
    def _summarize(self, text):
        pass

    @get_sents
    def summarize(self, text, sep='. ', end='.', top_n=0, percent=0):
        return sep.join(select(self._summarize(text), top_n, percent)) + end

    @get_sents
    def categorize(self, text, keywords, top_n=0, percent=1, threhold=None):
        summary = select(self._summarize(text), top_n, percent)
        txt = {word for sent in summary
               for word in sent}
        kw = {keywords} if isinstance(keywords, str) else set(keywords)
        if threhold:
            jaccard_simi = len(txt & kw) / len(txt | kw)
            return jaccard_simi >= threhold
        return kw < txt


class CosineSummarizer(BaseSummarizer):
    def _summarize(self, sents):
        return cosine_summarize(sents)


class FrequencySummarizer(BaseSummarizer):
    def _summarize(self, text):
        return freq_summarize(text)


cosine_categorize = partial(CosineSummarizer().categorize,
                            percent=0.3,
                            threhold=0.0015)
freq_categorize = partial(FrequencySummarizer().categorize,
                          percent=0.3,
                          threhold=0.0015)

# Demo video: https://www.youtube.com/watch?v=-OQ3l6lpzoc
