from __future__ import annotations


def build_prompt(question: str, context: str) -> str:
    if not context.strip():
        return (
            'You are a helpful assistant. The document context is empty, so answer based on your knowledge '
            'and clearly state that the answer is not supported by the retrieved documents.\n\n'
            f'Question: {question}'
        )

    return (
        'You are a helpful assistant. Answer the user question using the provided context only when the answer is supported by the documents. '
        'If the information is not present in the context, clearly say it is not available in the retrieved documents.\n\n'
        'Context:\n'
        f'{context}\n\n'
        f'Question: {question}'
    )