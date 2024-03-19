import streamlit as st
import os
from huggingface_hub import login
import torch
import transformers
import numpy as np
import pandas as pd
import json
import weaviate
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Benefits Q&A Chat")

# Replicate Credentials
with st.sidebar:
    st.title('"Benefits Q&A Chat"')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')


