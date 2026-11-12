from __future__ import annotations

from groq import Groq

from rag.config import GROQ_API_KEY, GROQ_MODEL


class GroqGenerator:
    """Generate answers using the Groq chat completions API."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or GROQ_MODEL
        self.api_key = api_key or GROQ_API_KEY
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def generate(self, prompt: str) -> str:
        if self.client is None:
            raise ValueError('GROQ_API_KEY is not set. Add your Groq API key to the .env file.')

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
        )
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError('Groq returned an empty response.')
        return answer