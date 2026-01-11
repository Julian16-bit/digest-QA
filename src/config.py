import os
import weaviate
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_weaviate_client():
    """
    Initialize and return Weaviate client.

    Tries to get credentials in this order:
    1. Streamlit secrets (for Streamlit Cloud deployment)
    2. Environment variables from .env file (for local development)
    """
    try:
        # Try Streamlit secrets first (for cloud deployment)
        api_key = st.secrets.get("WEAVIATE_API_KEY")
        url = st.secrets.get("WEAVIATE_URL")
    except (FileNotFoundError, KeyError):
        # Fall back to environment variables (for local development with .env)
        api_key = os.getenv("WEAVIATE_API_KEY")
        url = os.getenv("WEAVIATE_URL")

    if not api_key:
        raise ValueError("WEAVIATE_API_KEY not found in Streamlit secrets or environment variables")

    auth_config = weaviate.AuthApiKey(api_key=api_key)

    client = weaviate.Client(
        url=url,
        auth_client_secret=auth_config
    )

    return client
