import streamlit as st
import os
import weaviate
from sentence_transformers import SentenceTransformer
from openai import OpenAI

auth_config = weaviate.AuthApiKey(api_key="MutVi6yIYXH5xsoUlxhDH0O4GwO1aBCe1Jz0")

client = weaviate.Client(
  url="https://digest-data-english-i0rn0tsj.weaviate.network",
  auth_client_secret=auth_config
)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 100px'>Benefits Q&A Chat</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
    [data-testid="stSidebar"]{
    background-color: #8C9DB2;  
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    api_token = st.text_input("Enter your OpenAI API Token:", type='password')
    temperature_selection = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    top_p_selection = st.sidebar.slider('Top_p', min_value=0.0, max_value=1.0, value=1.0, step=0.05)

def create_prompt(query):
  model_name = 'sentence-transformers/all-MiniLM-L6-v2'
  vect_model = SentenceTransformer(model_name)
  query1 = query
  query_embedding = vect_model.encode(query1)
  response = (
  client.query
  .get("Digest2", ["content", "section_title", "doc_id", "section_chapter"])
  .with_hybrid(query=query1, vector=query_embedding)
  .with_additional(["score"])
  .with_limit(5)
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
    
  contents = [item['content'] for item in response['data']['Get']['Digest2']]
  formatted_text = '\n'.join([f"Supporting Text {i+1}: {item}" for i, item in enumerate(contents)])
  prompt = f"""
  As an AI assistant, specialized in question-answering tasks, your goal is to offer informative and accurate responses
  based on the provided context. I want you to imagine you're explaining things to someone in 8th grade, so keep your responses clear, simple, and easy to understand. 
  The provided context contains the principles applied in the Employment Insurance (EI) program, and the question is also related to the EI program.
  If you can't find the answer, just say 'I don't have an answer for this question.' Remember, be polite and concise in your responses.

  context: {formatted_text}
  Question: {query}
  Answer:
  """
  return prompt, results

def clear_chat_history():
    st.session_state.messages = []
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter your question here")
if user_input:
  prompt, results = create_prompt(user_input)
  gpt = OpenAI(api_key=api_token)
  completion = gpt.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": " As an AI assistant, specialized in question-answering tasks, your goal is to offer informative and accurate responses based on the provided context. I want you to imagine you're explaining things to someone in 8th grade, so keep your responses clear, simple, and easy to understand. The provided context contains the principles applied in the Employment Insurance (EI) program, and the question is also related to the EI program. If you can't find the answer, just say 'I don't have an answer for this question.' Remember, be polite and concise in your responses."},
    {"role": "user", "content": prompt}
  ],
    temperature=temperature_selection,
    top_p=top_p_selection
  )
  
  output = completion.choices[0].message
  content_output = output.content
  clean_output = content_output.replace("$", "\$")
  
  st.session_state.messages.append({"role": "user", "content": user_input})
  st.session_state.messages.append({"role": "assistant", "content": clean_output})
  
  st.chat_message("user").markdown(user_input)
  with st.chat_message("assistant"):
      st.markdown(clean_output)
  with st.expander("Click here to see the source"):
    st.write(results)
  
  #with col2:
    #with st.expander("Click here to see the source"):
      #st.write(results)
