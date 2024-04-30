import streamlit as st
import os
import torch
import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import pandas as pd

auth_config = weaviate.AuthApiKey(api_key="yxUlngVeQnPqTahq3vTuhBvW7rkMQLAJoSSH")

client = weaviate.Client(
  url="https://digest-english-t6pnsfgq.weaviate.network",
  auth_client_secret=auth_config
)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 100px'>AI-Powered Query Tool for Benefits🍁</h1>", unsafe_allow_html=True)

with st.sidebar:
    api_token = st.text_input("Enter your OpenAI API Token:", type='password')
    temperature_selection = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    top_p_selection = st.sidebar.slider('Top_p', min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    # grade_level = st.selectbox('Choose the level of complexity',('elementary school', 'middle school', 'high school', 'college' ))
    # st.write('You selected:', grade_level)

def create_prompt(query):
  model_name = 'sentence-transformers/all-MiniLM-L6-v2'
  vect_model = SentenceTransformer(model_name)
  reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  query1 = query
  query_embedding = vect_model.encode(query1)
  response = (
  client.query
  .get("Digest2", ["content", "section_title", "doc_id", "section_chapter"])
  .with_hybrid(query=query1, vector=query_embedding)
  .with_additional(["score"])
  .with_limit(20)
  .do()
  )
  
  results = []
  for item in response['data']['Get']['Digest2']:
    result = {
        'doc_id': item['doc_id'],
        'section_title': item['section_title'],
        'section_chapter': item['section_chapter'],
        'score': item['_additional']['score'],
        'content': item['content']
    }
    results.append(result)

  query_doc_pairs = [[query, res["content"]] for res in response["data"]["Get"]["Digest2"]]

  scores = reranker_model.predict(query_doc_pairs)
  print(scores)

  top_n = 5 ### Cap number of documents that are sent to LLM for RAG
  scores_cp = scores.tolist()
  documents = [pair[1] for pair in query_doc_pairs]
  content = ""
  content_display = []

  for _ in range(top_n):
    index = scores_cp.index(max(scores_cp))
    content += documents[index]
    content_display.append(documents[index])

    del documents[index]
    del scores_cp[index]

  content_set = set(content_display)
  doc_display = [docs for docs in results if docs['content'].strip().replace("Ã©", "é") in content_set]
  df = pd.DataFrame.from_dict(doc_display, orient='columns')
    
  prompt = f"""
  As an AI assistant specialized in question-answering tasks, your goal is to offer informative and accurate responses
  based on the provided context. If the answer cannot be found within the provided documents, respond with 'I don't have
  an answer for this question.' Be as concise and polite in your response as possible, and use simple language. The provided context contains the
  principles applied in the Employment Insurance (EI) program, and the question is also related to the EI program.

  Context: {content}
  Question: {query}
  Answer:
  """
  return prompt, df

def clear_chat_history():
    pass

user_input = st.text_input("", placeholder="What information are you looking for?", label_visibility="collapsed")

if user_input:
  prompt, df = create_prompt(user_input)
  gpt = OpenAI(api_key=api_token)
  completion = gpt.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "an AI assistant specialized in question-answering tasks, your goal is to offer informative and accurate responses only based on the provided context. If the answer cannot be found within the provided documents, respond with 'I don't have an answer for this question.' Be as concise and polite in your response as possible. "},
    {"role": "user", "content": prompt}
  ],
    temperature=temperature_selection,
    top_p=top_p_selection
  )
  
  output = completion.choices[0].message
  content_output = output.content
  clean_output = content_output.replace("$", "\$")
  
  st.table(df)

  if st.button("Generate summary"):
    st.write(clean_output)
          
