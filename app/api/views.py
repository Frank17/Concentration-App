from flask import request
from . import api_bp
from ._algo import lemmatizer
from ._summarizers import cosine_categorize

@api_bp.route('/summary', methods=['GET', 'POST'])
def summarize():
	if request.method == 'POST':
		f = request.form
		text = f['text']
		keywords = (lemmatizer.lemmatize(subject)
                    for subject in f['subjects'].split(','))
		return str(cosine_categorize(text, keywords))
	return 'POST me'
