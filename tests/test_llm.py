from types import SimpleNamespace

import pytest

from rag.generation.llm import GroqGenerator


def test_generate_requires_groq_api_key():
    generator = GroqGenerator(api_key='')

    with pytest.raises(ValueError, match='GROQ_API_KEY is not set'):
        generator.generate('Hello')


def test_generate_returns_groq_response():
    generator = GroqGenerator(model_name='test-model', api_key='test-key')
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='Hello from Groq'))]
                )
            )
        )
    )

    assert generator.generate('Hello') == 'Hello from Groq'
