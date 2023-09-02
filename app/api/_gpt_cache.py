from rapidfuzz import fuzz
from ._nltk_handler import STOPWORDS

import re
from datetime import datetime, timedelta
from typing import Callable


ALPHABET_REGEX = re.compile('[^a-z]+')
MIN = timedelta(0, 60)

def is_simi(text_a: str, text_b: str) -> bool:
    return fuzz.ratio(text_a, text_b) >= 75


def get_clean_text(text: str) -> str:
    return ' '.join(
        [word for word in text.split()
         if ALPHABET_REGEX.sub('', word.lower()) not in STOPWORDS]
    )


class judgment_cache:
    """A simple TTL cache that saves URLs with
       GPT-made judgements

       Item: {url: (judgment, start_time, keywords, text)}
    """
    
    def __init__(self, requester: Callable):
        self._cache: dict = {}
        self.requester: Callable = requester
        
    
    def __call__(self, url: str, text: str, keywords: list[str]) -> str:
        if (
            (record := self._cache.get(url)) and
            datetime.now() - record[1] < MIN and
            keywords == record[2] and
            is_simi(record[3], get_clean_text(text))
        ):
            return record[0]
        
        if len(self._cache) % 50 == 0:
            self._cache.clear()
            
        self._cache[url] = (
            self.requester(url, text, keywords),  # judgment
            datetime.now(),  # start time
            keywords,
            text
        )
        return self._cache[url][0]