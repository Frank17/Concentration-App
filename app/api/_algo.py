from __future__ import annotations

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import linear_kernel
from igraph import Graph
from ._nltk_handler import STOPWORDS, WNL

from collections import Counter
import re


INVALID_CHAR_REGEX = re.compile('[^a-zA-Z’\'\s]+')


def get_sents(text: str) -> list[str]:
    """Return a list of cleaned sentences with lemmatized words
    """
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
    """Return a summary of the text using cosine similarity algorithm
    """
    sents = get_sents(text)
    tfidf_pl = make_pipeline(
        # Hashing vec. is employed since it is more memory-efficient.
        # When n_features is set to 2 ** 17, this approach is ≈4.5 times
        # slower than using TfidfVectorizer directly, which is acceptable.
        HashingVectorizer(lowercase=False, n_features=2**17, norm=None),
        TfidfTransformer()
    )
    tfidf_mtx = tfidf_pl.fit_transform(sents)
    cos_simi_mtx = linear_kernel(tfidf_mtx)
    # Alternatively, ... = (tfidf_mtx * tfidf_mtx.T).A also works

    cos_simi_gph = Graph.Weighted_Adjacency(cos_simi_mtx)
    scores = cos_simi_gph.pagerank(weights="weight")
    return dict(zip(sents, scores))


def freq_summarize(text: str) -> dict[str, float]:
    """Return a summary of the text using frequency analysis
    """
    sents = get_sents(text)
    abs_freqs = Counter(word for sent in sents for word in sent)
    max_count = max(abs_freqs.values())
    rel_freqs = {word: count / max_count for word, count in abs_freqs.items()}
    return {
        ' '.join(sent): sum(rel_freqs.get(word, 0) for word in sent)
        for sent in sents
    }
