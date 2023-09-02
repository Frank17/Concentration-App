import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from functools import lru_cache


STOPWORDS_EXT = {
    "i’ve", "i’m", "we’ve", "you’ve", "she’ve", "he’ve", "they’ve",
    "haven’t", "hasn’t",
    "don’t", "does’t", "didn’t",
    "wouldn’t", "won’t", "without",
    "i've", "i'm", "we've", "you've", "she've", "he've", "they've",
    "haven't", "hasn't"
    "don't", "does't", "didn't",
    "wouldn't", "won't", "without",
    "would", "need", "like", "also",
    "never", "rarely", "sometimes", "often", "oftentimes", "always",
}

try:
    stopwords.words
except LookupError:
    nltk.download('stopwords')
    
try:
    WordNetLemmatizer().lemmatize('')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english')) | STOPWORDS_EXT
WNL = lru_cache(2 ** 7)(WordNetLemmatizer().lemmatize)