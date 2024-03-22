import streamlit as st
import os
import weaviate
from sentence_transformers import SentenceTransformer
from openai import OpenAI

auth_config = weaviate.AuthApiKey(api_key="uokLNfAvageSXij8kUuTlh53DPBz3HMG5Rc5")

client = weaviate.Client(
  url="https://digest-data-2-hukgw816.weaviate.network",
  auth_client_secret=auth_config
)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 100px'>Benefits Q&A Chat</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with st.sidebar:
    api_token = st.text_input("Enter your OpenAI API Token:", type='password')

def create_prompt(query):
  model_name = 'sentence-transformers/all-MiniLM-L6-v2'
  vect_model = SentenceTransformer(model_name)
  query1 = query
  query_embedding = vect_model.encode(query1)
  response = (
  client.query
  .get("Digest2", ["content", "section_title", "doc_id"])
  .with_hybrid(query=query1, vector=query_embedding)
  .with_additional(["score"])
  .with_limit(5)
  .do()
  )
  
  results = []
  for item in response['data']['Get']['Digest2']:
    result = {
        'doc_id': item['doc_id'],
        'score': item['_additional']['score'],
        'content': item['content']
    }
    results.append(result)
    
  contents = [item['content'] for item in response['data']['Get']['Digest2']]
  formatted_text = '\n'.join([f"Supporting Text {i+1}: {item}" for i, item in enumerate(contents)])
  prompt = f"""
  As an AI assistant specialized in question-answering tasks, your goal is to offer informative and accurate responses
  based on the provided context. If the answer cannot be found within the provided documents, respond with 'I don't have
  an answer for this question.' Be as concise and polite in your response as possible. The provided context contains the
  principles applied in the Employment Insurance (EI) program, and the question is also related to the EI program.

  context: {formatted_text}
  Question: {query}
  Answer:
  """
  return prompt, results

user_input = st.chat_input("Enter your question here")
if user_input:
  prompt, results = create_prompt(user_input)
  gpt = OpenAI(api_key=api_token)

  completion = gpt.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "an AI assistant specialized in question-answering tasks, your goal is to offer informative and accurate responses only based on the provided context. If the answer cannot be found within the provided documents, respond with 'I don't have an answer for this question.' Be as concise and polite in your response as possible. "},
    {"role": "user", "content": prompt}
  ]
  )
  
  output = completion.choices[0].message

  with col1:
    st.write(f"Chatbot answer to: {user_input}")
    st.write(output.content)
    
  st.markdown("<hr>", unsafe_allow_html=True)
  
  with col2:
    with st.expander("Click here to see the source"):
      st.write(results)
