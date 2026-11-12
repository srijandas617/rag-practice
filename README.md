# Production RAG Pipeline

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline using:

- PDF ingestion via PyMuPDF
- Text cleaning and chunking
- SentenceTransformers embeddings
- ChromaDB vector storage
- Optional semantic + keyword retrieval
- Groq API-based generation with `llama-3.3-70b-versatile`
- Streamlit UI for document upload and querying

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Get an API key from the Groq Console. The app raises an explicit error when you submit a question while `GROQ_API_KEY` is blank.

4. Run the app:

```bash
py -m streamlit run .\app.py
```

## Environment configuration

The app reads configuration from `.env`. Set `GROQ_API_KEY` and any optional settings in that file; do not commit `.env` because it may contain secrets.

## Deploy to Render

1. Push this project to a GitHub repository and create a Render Blueprint from that repository. Render will use `render.yaml` to create the web service and its persistent disk.
2. Set `GROQ_API_KEY` in the Render dashboard when prompted. Do not commit the key or put it in `render.yaml`.
3. Deploy the service. Render runs Streamlit on its assigned port and checks `/_stcore/health`.

The Blueprint uses a Starter web service because Render persistent disks require a paid instance. ChromaDB data and uploaded PDFs are stored on the mounted disk at `/var/data`, so they persist across deploys. The disk is attached to one instance; multiple instances should not share this local ChromaDB store.

## Chat

The Streamlit interface displays questions and answers as a conversation and provides a **Clear conversation** button. Chat history is kept in Streamlit session state for the current session; it is not saved across sessions. Each question is currently answered independently using document retrieval, so earlier messages are displayed but are not passed to the model as context.