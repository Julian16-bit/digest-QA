import streamlit as st
from src.config import get_weaviate_client
from src.retrieval import create_prompt, initialize_models
from src.llm import get_llm_response

# Initialize models on startup
with st.spinner("Loading models..."):
    initialize_models()

# Initialize Weaviate client
client = get_weaviate_client()

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 100px'>Canada Benefits Q&A Chat</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
    [data-testid="stSidebar"]{
    background-color: #8C9DB2;  
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    api_token = st.text_input("Enter your OpenAI API Token:", type='password')
    temperature_selection = st.slider('Temperature', min_value=0.0, max_value=2.0, value=1.0, step=0.05)
    top_p_selection = st.slider('Top_p', min_value=0.0, max_value=1.0, value=1.0, step=0.05)

def clear_chat_history():
    st.session_state.messages = []

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
user_input = st.chat_input("Enter your question here")
if user_input:
    # Get retrieval results and prompt
    prompt, results = create_prompt(user_input, client)

    # Get LLM response
    clean_output = get_llm_response(api_token, prompt, temperature_selection, top_p_selection)

    # Update session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": clean_output})

    # Display messages
    st.chat_message("user").markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(clean_output)
    with st.expander("Click here to see the source"):
        st.write(results)
