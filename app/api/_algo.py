from __future__ import annotations

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import linear_kernel
from igraph import Graph
from ._nltk_handler import STOPWORDS, WNL

from collections import Counter


INVALID_CHAR_REGEX = re.compile('[^a-zA-Z’\'\s]+')


def get_sents(text: str) -> list[str]:
    sents = re.split('[.!?] ', re.sub('([—-]+|;|:)', ' ', text))
    cleaned_sents = []
    add_sent = cleaned_sents.append

    for sent in sents:
        sent = INVALID_CHAR_REGEX.sub('', sent)
        if sent:
            add_sent(' '.join(
                [WNL(word) for word in sent.lower().split()
                 if word not in STOPWORDS]
            ))
    return cleaned_sents


def cos_summarize(text: str) -> dict[str, float]:
    sents = get_sents(text)
    tfidf_vec = TfidfVectorizer(stop_words=None)
    tfidf_mtx = tfidf_vec.fit_transform(sents)
    cos_simi_mtx = tfidf_mtx * tfidf_mtx.T
    
    cos_simi_gph = Graph.Weighted_Adjacency(cos_simi_mtx.A)
    scores = cos_simi_gph.pagerank(weights="weight")
    return dict(zip(sents, scores))


def freq_summarize(sents: list[list[str]]) -> dict[str, float]:
    abs_freqs = Counter(word for sent in sents for word in sent)
    max_count = max(abs_freqs.values())
    rel_freqs = {word: count / max_count for word, count in abs_freqs.items()}
    return {
        ' '.join(sent): sum(rel_freqs.get(word, 0) for word in sent)
        for sent in sents
    }
