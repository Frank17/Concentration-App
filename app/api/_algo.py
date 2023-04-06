from __future__ import annotations
from itertools import permutations
from collections import Counter
from nltk.cluster.util import cosine_distance
from heapq import nlargest
import re
import numpy as np
from igraph import Graph

from functools import wraps, lru_cache
from ._handler import STOPWORDS_EXT, download_nltk

_stopwords, lemmatizer = download_nltk()
_stopwords.extend(STOPWORDS_EXT)
_lemmatize = lru_cache(256)(lemmatizer.lemmatize)
STOPWORDS = set(_stopwords)

FALLBACK_MAX = 40
FALLBACK_LEVEL = 10
_curr_fb = FALLBACK_MAX + FALLBACK_LEVEL

INVALID_CHAR_REGEX = re.compile('[^a-zA-Z’\s]+')


def get_sents(f):
    @wraps(f)
    def text2sents(self, text, *args, **kwargs):
        sents = re.split('[.!?] ', re.sub('([—-]+|;|:)', ' ', text))
        cleaned_sents = []
        add_sent = cleaned_sents.append

        for sent in sents:
            sent = INVALID_CHAR_REGEX.sub('', sent)
            if sent:
                add_sent(
                    tuple([
                        _lemmatize(word) for word in sent.lower().split()
                        if word not in STOPWORDS and len(word) > 3
                    ]))
        return f(self, cleaned_sents, *args, **kwargs)

    return text2sents


def _get_cosine_similarity(sent1: list[str], sent2: list[str]) -> float:
    words = list(set(sent1 + sent2))

    def get_vector(sent):
        vec = [0] * len(words)
        for word in sent:
            vec[words.index(word)] += 1
        return vec

    with np.errstate(invalid='ignore'):
        return 1 - cosine_distance(get_vector(sent1), get_vector(sent2))


def _get_similarity_matrix(
        sents: list[list[str]]) -> np.ndarray[np.ndarray[float]]:
    sent_n = len(sents)
    simi_mtx = np.zeros((sent_n, sent_n))

    for i, j in permutations(range(sent_n), 2):
        simi_mtx[i][j] = _get_cosine_similarity(sents[i], sents[j])

    return simi_mtx


def cosine_summarize(sents: list[list[str]]) -> dict[str, float]:
    simi_mtx = _get_similarity_matrix(sents)
    scores = Graph.Adjacency(simi_mtx).pagerank()
    return dict(zip(sents, scores))


def freq_summarize(sents):
    abs_freqs = Counter(word for sent in sents for word in sent)
    max_count = max(abs_freqs.values())
    rel_freqs = {word: count / max_count for word, count in abs_freqs.items()}
    return {
        ' '.join(sent): sum(rel_freqs.get(word, 0) for word in sent)
        for sent in sents
    }


def select(scored_dict, top_n: int = 0, percent: float = 0):
    n = max(top_n, round(percent * len(scored_dict)))
    return nlargest(n, scored_dict, key=scored_dict.get)
