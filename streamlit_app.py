import streamlit as st
import os
import weaviate
from sentence_transformers import SentenceTransformer

auth_config = weaviate.AuthApiKey(api_key="Dj76ptxASwSdQuptoSrJnUzsSxnlnxoK7DSK")

client = weaviate.Client(
  url="https://digest-data-2-vccdanml.weaviate.network",
  auth_client_secret=auth_config
)

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
vect_model = SentenceTransformer(model_name)

query = 'Can I still receive benefits as a substitute teacher?'
query_embedding = vect_model.encode(query)

response = (
  client.query
  .get("Digest2", ["content", "section_title", "doc_id"])
  .with_hybrid(query=query, vector=query_embedding)
  .with_additional(["score"])
  .with_limit(5)
  .do()
)

st.set_page_config(page_title="Benefits Q&A Chat")

# Replicate Credentials
with st.sidebar:
    st.title('"Benefits Q&A Chat"')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')


