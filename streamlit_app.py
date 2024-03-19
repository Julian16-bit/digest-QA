import streamlit as st

conn = st.connection(
    "weaviate",
    type=WeaviateConnection,
    url=os.getenv("https://digest-data-2-vccdanml.weaviate.network"),
    api_key=os.getenv("Dj76ptxASwSdQuptoSrJnUzsSxnlnxoK7DSK"),
   )

st.set_page_config(page_title="Benefits Q&A Chat")

# Replicate Credentials
with st.sidebar:
    st.title('"Benefits Q&A Chat"')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')


