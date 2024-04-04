import streamlit as st
import os
import torch
import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

auth_config = weaviate.AuthApiKey(api_key="9tqoCx5yjyrLpRAq288U6l3385oOeW22t3vo") 
 
client = weaviate.Client(  
  url="https://french-digest-co42fmv3.weaviate.network",
  auth_client_secret=auth_config 
)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 100px'>Prestation Q&A Chat</h1>", unsafe_allow_html=True)

with st.sidebar:
    api_token = st.text_input("Entrez votre cle API OpenAI ici:", type='password')
    temperature_selection = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    top_p_selection = st.sidebar.slider('Top_p', min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    # grade_level = st.selectbox('Choose the level of complexity',('elementary school', 'middle school', 'high school', 'college' ))
    # st.write('You selected:', grade_level)

def create_prompt(query):
  tokenizer = AutoTokenizer.from_pretrained('camembert-base')
  reranker_model = SentenceTransformer('antoinelouis/biencoder-electra-base-french-mmarcoFR')
  query1 = query
  query_embedding = tokenizer(query)
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

  query_doc_pairs = [[],[]]
  
  query_doc_pairs[0] = [query for res in response["data"]["Get"]["Digest2"]]
  query_doc_pairs[1] = [res["content"] for res in response["data"]["Get"]["Digest2"]]

  q_embeddings = reranker_model.encode(query_doc_pairs[0], normalize_embeddings=True)
  d_embeddings = reranker_model.encode(query_doc_pairs[1], normalize_embeddings=True)

  scores = (q_embeddings @ d_embeddings.T)[0]
  print(scores)

  top_n = 5 ### Cap number of documents that are sent to LLM for RAG
  scores_cp = scores.tolist()
  documents = query_doc_pairs[1]
  content = ""

  for _ in range(top_n):
    index = scores_cp.index(max(scores_cp))
    content += documents[index]

    del documents[index]
    del scores_cp[index]
    
  prompt = f"""
  En tant qu'assistant IA spécialisé dans les tâches de réponse aux questions, votre objectif est d'offrir des réponses informatives et précises basées sur le contexte fourni. Si la réponse ne peut pas être trouvée dans les documents fournis, répondez par « Je n'ai pas de réponse à cette question ». Soyez aussi concis et poli que possible dans votre réponse et utilisez un langage simple. Le contexte fourni contient les principes appliqués dans le programme d'assurance-emploi (AE), et la question est également liée au programme d'AE.

  Contexte: {content}
  Question: {query}
  Réponse:
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
    {"role": "system", "content": "En tant qu'assistant IA spécialisé dans les tâches de réponse aux questions, votre objectif est de proposer des réponses informatives et précises uniquement basées sur le contexte fourni. Si la réponse ne peut pas être trouvée dans les documents fournis, répondez par « Je n'ai pas de réponse à cette question ». Soyez aussi concis et poli que possible dans votre réponse."},
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
          
