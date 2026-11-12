import os
from pathlib import Path

import streamlit as st

from rag.config import UPLOAD_DIR
from rag.pipeline.ingestion_pipeline import IngestionPipeline
from rag.pipeline.query_pipeline import QueryPipeline


st.set_page_config(page_title='Production RAG Pipeline', page_icon='📚', layout='wide')


@st.cache_resource
def get_ingestion_pipeline():
    return IngestionPipeline()


@st.cache_resource
def get_query_pipeline():
    return QueryPipeline()


def main():
    st.title('Production RAG Pipeline')
    st.caption('Upload a PDF, ingest it into ChromaDB, and ask questions grounded in the indexed documents.')

    uploaded_file = st.file_uploader('Upload PDF', type=['pdf'])

    if uploaded_file is not None:
        save_dir = UPLOAD_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.read())
        st.success(f'File saved to {file_path}')

        if st.button('Ingest document'):
            pipeline = get_ingestion_pipeline()
            result = pipeline.process_file(str(file_path))
            st.success(f'Ingestion complete. {result["chunks_count"]} chunks stored in ChromaDB.')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if st.button('Clear conversation'):
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])
            if message['role'] == 'assistant' and message.get('sources'):
                with st.expander('Sources'):
                    for item in message['sources']:
                        st.markdown(f'- {item["source"]} (page {item["page"]})')

    question = st.chat_input('Ask a question about the uploaded document set')
    if question:
        st.session_state.messages.append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.write(question)

        pipeline = get_query_pipeline()
        with st.chat_message('assistant'):
            with st.spinner('Searching your documents and generating an answer...'):
                result = pipeline.answer(question, top_k=4)
            st.write(result['answer'])

            sources = result.get('sources', [])
            if sources:
                with st.expander('Sources'):
                    for item in sources:
                        st.markdown(f'- {item["source"]} (page {item["page"]})')

        st.session_state.messages.append({
            'role': 'assistant',
            'content': result['answer'],
            'sources': sources,
        })


if __name__ == '__main__':
    main()