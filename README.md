# Benefits Q&A Chat

An AI-powered question-answering system for Employment Insurance (EI) program information using RAG (Retrieval-Augmented Generation).

## Features

- **Hybrid Search**: Combines semantic and keyword search using Weaviate vector database
- **Reranking**: Uses cross-encoder models to improve retrieval quality
- **Interactive Chat**: Streamlit-based chat interface with conversation history
- **Source Citations**: View the source documents used to generate each answer

## Prerequisites

- Python 3.8+
- OpenAI API key
- Weaviate database access

## Usage

Run the Streamlit Cloud application at this URL (requires valid OpenAI API key):
https://digest-app-default-rerank.streamlit.app

The app will:
1. Load embedding and reranking models on startup
2. Prompt you to enter your OpenAI API token in the sidebar

## Project Structure

```
digest-QA/
├── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── config.py             # Weaviate client configuration
│   ├── retrieval.py          # Vector search and reranking
│   └── llm.py                # OpenAI integration
├── data_scrape/
│   └── page_scrape.py        # Web scraping utilities
└── .env                      # Environment variables (not committed)
```

## Configuration

Adjust parameters in `src/retrieval.py`:
- `INITIAL_RETRIEVAL_LIMIT`: Number of documents to retrieve (default: 20)
- `TOP_N_RESULTS`: Number of documents to use for RAG (default: 5)
- `EMBEDDING_MODEL`: Sentence transformer model
- `RERANKER_MODEL`: Cross-encoder reranking model
