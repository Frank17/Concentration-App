from flask import request
from . import api_bp
from ._algo import lemmatizer
from ._summarizers import CosineSummarizer

PERCENT = 0.4
THRESHOLD = 0.0015
_lemmatize = lemmatizer.lemmatize


@api_bp.route('/summary', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        f = request.form
        text = f['text']
        keywords = f['subjects'].split(',')
        keywords = {_lemmatize(kw) for kw in keywords}
        cos_summarizer = CosineSummarizer()
        return str(cos_summarizer.categorize(text, keywords, PERCENT, THRESHOLD))
    return 'POST Me'