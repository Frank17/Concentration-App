import tiktoken
import openai
from ._algo import cos_summarize
from ._gpt_cache import judgment_cache

PROMPT = ('You will be provided with an article from the website {url}. '
          'It has been cleaned to remove stopwords. '
          'Based on this article and your knowledge of the website, '
          'is the article related to {keywords}? Output only yes or no.')

GPT3_ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo')

openai.api_key = os.getenv('OPENAI_API_KEY')


@judgment_cache
def get_gpt_judgment(url: str, text: str, keywords: str):
    def get_token_n(text: str) -> int:
        return len(GPT3_ENCODING.encode(text))
        
    if len(keywords) == 1:
        keywords = keywords[0]
    else:
        keywords = f'{", ".join(keywords[:-1])}, or {keywords[-1]}'
        
    req_prompt = PROMPT.format(url=url, keywords=keywords)
    max_text_token_n = 4090 - get_token_n(req_prompt)
    
    if get_token_n(text) > max_text_token_n:
        summary = cos_summarize(text)
        summary = sorted(summary, key=summary.get)
        text = ''
        while summary:
            next_sent = summary.pop()
            if get_token_n(text + next_sent) > max_text_token_n:
                break
            text += ' ' + next_sent

    try:
        return openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'system', 'content': req_prompt},
                      {'role': 'user', 'content': text}],
            temperature=0,
            max_tokens=2
        )['choices'][0]['message']['content'].lower()
    except openai.error.RateLimitError:
        return 'lim'
