"""
Vector search and retrieval module for Employment Insurance Q&A system.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Configuration constants
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
INITIAL_RETRIEVAL_LIMIT = 20
TOP_N_RESULTS = 5
COLLECTION_NAME = "Digest2"

# Prompt template
SYSTEM_PROMPT = """As an AI assistant specialized in question-answering tasks, your goal is to offer informative and accurate responses
based on the provided context. If the answer cannot be found within the provided documents, respond with 'I don't have
an answer for this question.' Be as concise and polite in your response as possible, and use simple language. The provided context contains the
principles applied in the Employment Insurance (EI) program, and the question is also related to the EI program.

Context: {context}
Question: {query}
Answer:
"""

# Global model instances
_embedding_model = None
_reranker_model = None

logger = logging.getLogger(__name__)


def initialize_models():
    """
    Initialize all models at startup to avoid lazy loading delays.
    Call this function when the application starts.
    """
    global _embedding_model, _reranker_model

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    if _reranker_model is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker_model = CrossEncoder(RERANKER_MODEL)

    logger.info("All models loaded successfully")


def get_embedding_model():
    """Get or initialize the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_reranker_model():
    """Get or initialize the reranker model."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def retrieve_documents(query, client, limit=INITIAL_RETRIEVAL_LIMIT):
    """
    Retrieve documents from Weaviate using hybrid search.

    Args:
        query: User query string
        client: Weaviate client instance
        limit: Number of documents to retrieve

    Returns:
        list: List of document dictionaries with metadata

    Raises:
        Exception: If Weaviate query fails
    """
    try:
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(query)

        response = (
            client.query
            .get(COLLECTION_NAME, ["content", "section_title", "doc_id", "section_chapter"])
            .with_hybrid(query=query, vector=query_embedding)
            .with_additional(["score"])
            .with_limit(limit)
            .do()
        )

        if not response.get('data', {}).get('Get', {}).get(COLLECTION_NAME):
            logger.warning(f"No results found for query: {query}")
            return []

        results = []
        for item in response['data']['Get'][COLLECTION_NAME]:
            result = {
                'doc_id': item['doc_id'],
                'section_title': item['section_title'],
                'section_chapter': item['section_chapter'],
                'score': item['_additional']['score'],
                'content': item['content']
            }
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise


def rerank_documents(query, documents, top_n=TOP_N_RESULTS):
    """
    Rerank documents using cross-encoder model.

    Args:
        query: User query string
        documents: List of document dictionaries
        top_n: Number of top documents to return

    Returns:
        list: Top N reranked documents
    """
    if not documents:
        return []

    reranker_model = get_reranker_model()
    query_doc_pairs = [[query, doc["content"]] for doc in documents]

    scores = reranker_model.predict(query_doc_pairs)

    # Get top N indices efficiently using argsort
    top_indices = np.argsort(scores)[::-1][:top_n]

    # Return documents in reranked order
    reranked_docs = [documents[idx] for idx in top_indices]

    return reranked_docs


def create_prompt(query, client):
    """
    Create a prompt by retrieving and reranking documents.

    Args:
        query: User's question
        client: Weaviate client instance

    Returns:
        tuple: (prompt string, list of top reranked document dictionaries)
    """
    try:
        # Retrieve initial documents
        documents = retrieve_documents(query, client)

        if not documents:
            return f"Question: {query}\nAnswer:", []

        # Rerank documents
        top_documents = rerank_documents(query, documents)

        # Combine content from top documents
        context = "\n\n".join([doc['content'] for doc in top_documents])

        # Create prompt
        prompt = SYSTEM_PROMPT.format(context=context, query=query)

        return prompt, top_documents

    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise