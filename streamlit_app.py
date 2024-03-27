import streamlit as st
import os
import torch
import weaviate
import statistics
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from evaluate import load

auth_config = weaviate.AuthApiKey(api_key="uokLNfAvageSXij8kUuTlh53DPBz3HMG5Rc5")

client = weaviate.Client(
  url="https://digest-data-2-hukgw816.weaviate.network",
  auth_client_secret=auth_config
)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 100px'>Benefits Q&A Chat</h1>", unsafe_allow_html=True)

with st.sidebar:
    api_token = st.text_input("Enter your OpenAI API Token:", type='password')
    temperature_selection = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    top_p_selection = st.sidebar.slider('Top_p', min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    # grade_level = st.selectbox('Choose the level of complexity',('elementary school', 'middle school', 'high school', 'college' ))
    # st.write('You selected:', grade_level)

# query_response_pairs = []
# def save_pair(query, answer):
  # global query_response_pairs
  # query_response_pairs.append([query, answer])

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

  for _ in range(top_n):
    index = scores_cp.index(max(scores_cp))
    content += documents[index]

    del documents[index]
    del scores_cp[index]
    
  prompt = f"""
  As an AI assistant specialized in question-answering tasks, your goal is to offer informative and accurate responses
  based on the provided context. If the answer cannot be found within the provided documents, respond with 'I don't have
  an answer for this question.' Be as concise and polite in your response as possible, and use simple language. The provided context contains the
  principles applied in the Employment Insurance (EI) program, and the question is also related to the EI program.

  Context: {content}
  Question: {query}
  Answer:
  """
  return prompt, results

def clear_chat_history():
    st.session_state.messages = []

def evaluation(query_response_pairs):
  ### Compute the semantic similarity score using cross encoder
  eval_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  similarity = eval_model.predict(query_response_pairs)


  ### Compute the precision, recall, and f score using bert_score model
  bertscore = load("bertscore")
  query = [pair[0] for pair in query_response_pairs]
  answer = [pair[1] for pair in query_response_pairs]
  results = bertscore.compute(predictions=answer, references=query, lang="en", num_layers=2) # Research says the correctness is the best when using second layer ouput as embeddings
  
  return statistics.mean(similarity), statistics.mean(results['precision']), statistics.mean(results['recall']), statistics.mean(results['f1'])


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
    {"role": "system", "content": "an AI assistant specialized in question-answering tasks, your goal is to offer informative and accurate responses only based on the provided context. If the answer cannot be found within the provided documents, respond with 'I don't have an answer for this question.' Be as concise and polite in your response as possible. "},
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

  # save_pair(user_input, clean_output)
      
  st.chat_message("user").markdown(user_input)
  with st.chat_message("assistant"):
      st.markdown(clean_output)
  with st.expander("Click here to see the source"):
    st.write(results)

if st.sidebar.button('Get Evaluation Metric'):
  with st.sidebar:
    if len(st.session_state.messages) == 0:
      st.write("No messages to evaluate!")
    else:
      query_response= []
      for message in st.session_state.messages:
        query_response.append(message["content"])
      query_response_pairs = []
      for i in range(0, len(query_response), 2):
        pair = [query_response[i], query_response[i + 1]]
        query_response_pairs.append(pair)
        
      similarity, precision, recall, f1 = evaluation(query_response_pairs)
      st.write(f"The similarity score is: {round(similarity, 2)}")
      st.write(f"The precision score is: {round(precision, 3)}")
      st.write(f"The recall score is: {round(recall, 3)}")
      st.write(f"The f1 score is: {round(f1, 3)}")
          
