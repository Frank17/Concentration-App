from flask import request
from . import api_bp
from ._nltk_handler import WNL
from ._judge import get_judgment

import json


@api_bp.route('/summary', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        f = request.form
        text: str = f['text']
        url: str = f['url']
        keywords: list[str] = [
            WNL(kw) for kw
            in json.loads(f['subjects'])
        ]
        mode: str = f['mode']
        return get_judgment(url, text, keywords, mode)
    
    return 'POST Me'