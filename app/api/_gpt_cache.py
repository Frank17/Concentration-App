from rapidfuzz import fuzz
from ._nltk_handler import STOPWORDS

import re
from datetime import datetime, timedelta
from typing import Callable


ALPHABET_REGEX = re.compile('[^a-z]+')
MIN = timedelta(0, 60)


def is_simi(text_a: str, text_b: str) -> bool:
    """Return whether text_a and text_b are similar
    """
    return fuzz.ratio(text_a, text_b) >= 75


def get_clean_text(text: str) -> str:
    """Get the stopwords-free version of the text
    """
    return ' '.join(
        [word for word in text.split()
         if ALPHABET_REGEX.sub('', word.lower()) not in STOPWORDS]
    )


class judgment_cache:
    """Keeps track of previous records of judgments to avoid repetitive requests

       Record format: {url: (judgment, start_time, keywords, text)}
    """
    
    def __init__(self, requester: Callable):
        self._cache: dict = {}
        self.requester: Callable = requester
        
    
    def __call__(self, url: str, text: str, keywords: list[str]) -> str:
        if (
            (record := self._cache.get(url)) and  # record exists
            datetime.now() - record[1] < MIN and  # record was saved within 1 min
            keywords == record[2] and  # keywords match
            is_simi(record[3], get_clean_text(text))  # similar article
        ):
            return record[0]
        
        # Clear the cache whenever there are 30 records
        if len(self._cache) % 30 == 0:
            self._cache.clear()
        
        # Send the request and save the new record
        self._cache[url] = (
            self.requester(url, text, keywords),  # judgment
            datetime.now(),  # start time
            keywords,
            text
        )
        return self._cache[url][0]