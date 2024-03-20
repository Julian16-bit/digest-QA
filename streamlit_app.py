import streamlit as st
import os
import weaviate
from sentence_transformers import SentenceTransformer
from openai import OpenAI

auth_config = weaviate.AuthApiKey(api_key="Dj76ptxASwSdQuptoSrJnUzsSxnlnxoK7DSK")

client = weaviate.Client(
  url="https://digest-data-2-vccdanml.weaviate.network",
  auth_client_secret=auth_config
)

st.title("Chatbot with GPT-3.5 Turbo")
st.write("Enter your question below:")

user_input = st.chat_input("Your question")

with st.sidebar:
    st.title('Benefits Q&A Chat')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    api_token = st.text_input("Enter your OpenAI API Token:", "")

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
  return prompt

if user_input:
  prompt = create_prompt(user_input)
  gpt = OpenAI(api_key=api_token)

  completion = gpt.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "an AI assistant specialized in question-answering tasks, your goal is to offer informative and accurate responses only based on the provided context. If the answer cannot be found within the provided documents, respond with 'I don't have an answer for this question.' Be as concise and polite in your response as possible. "},
    {"role": "user", "content": prompt}
  ]
  )
  
  output = completion.choices[0].message
  
  st.write("Chatbot's response:")
  st.write(output)


