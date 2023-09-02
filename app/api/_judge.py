from ._strict_judge import get_strict_judgment
from ._gpt_judge import get_gpt_judgment

def get_judgment(url: str, text: str, keywords: list[str], mode: str) -> str:
    if mode == 'strict':
        return get_strict_judgment(text, set(keywords))
    elif mode == 'gpt':
        return get_gpt_judgment(url, text, keywords)
    return 'no'