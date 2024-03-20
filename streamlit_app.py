import streamlit as st
import os
import weaviate

auth_config = weaviate.auth(api_key="Dj76ptxASwSdQuptoSrJnUzsSxnlnxoK7DSK")

client = weaviate.Client(
  url="https://digest-data-2-vccdanml.weaviate.network",
  auth_client_secret=auth_config
)

st.set_page_config(page_title="Benefits Q&A Chat")

# Replicate Credentials
with st.sidebar:
    st.title('"Benefits Q&A Chat"')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')


