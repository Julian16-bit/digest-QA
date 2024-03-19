import streamlit as st
import os

st.set_page_config(page_title="Benefits Q&A Chat")

# Replicate Credentials
with st.sidebar:
    st.title('"Benefits Q&A Chat"')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    if 'HUGGINGFACE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        huggingface_api = st.secrets['HUGGINGFACE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Hugging Face API token:', type='password')
        if not (replicate_api.startswith('hf_') and len(replicate_api)==37):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['HUGGINGFACE_API_TOKEN'] = huggingface_api

