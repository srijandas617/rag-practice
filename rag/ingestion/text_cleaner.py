import re


class TextCleaner:
    """Clean extracted PDF text before chunking."""

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ''

        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n{2,}', '\n\n', text)
        return text.strip()