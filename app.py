from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import streamlit as st
import time

st.set_page_config(page_title="TeleConnect Companion", page_icon="📶", layout="centered")

load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [{
        "user": "Hi", 
        "bot": "Welcome to TeleConnect, how can I help you?"
    }]

# Sidebar 
st.sidebar.markdown(
    """
    <style>
    .gif-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("📶 Welcome to TeleConnect")

# Adding a live Wi-Fi GIF to the sidebar
st.sidebar.markdown(
    """
    <div class="gif-container">
        <img src="https://media.tenor.com/BDzwAyYBqUsAAAAj/wifi-strong-connection.gif" alt="Wi-Fi Signal">
    </div>
    """,
    unsafe_allow_html=True
)

# Navbar with dark blue color, hover effect, and curved edges
st.markdown(
    """
    <style>
    .navbar {
        background-color: #00008B;
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        border-radius: 15px;
        transition: background-color 0.3s, color 0.3s;
    }
    .navbar:hover {
        background-color: #007FFF;
        color: #E0E0E0;
    }
    </style>
    <div class="navbar">TeleConnect Companion</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stTextInput input {
        background-color: #444;
        color: #fff;
    }
    .stButton button {
        background-color: #444;
        color: #fff;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        padding: 10px;
    }
    .message-container {
        display: flex;
        margin-bottom: 10px;
    }
    .user-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
        align-items: center;
    }
    .message-bubble {
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        position: relative;
    }
    .user-bubble {
        background-color: #0000FF; /* Blue */
        color: white;
        text-align: right;
    }
    .bot-bubble {
        background-color: white; /* White background for bot */
        color: black;
        text-align: left;
    }
    .message-text {
        margin-right: 8px;
    }
    .bot-container {
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .emoji {
        margin-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def load_data():
    loader = CSVLoader(file_path="qna.csv", source_column="ï»¿prompt")
    data = loader.load()
    return data

def create_embeddings(data):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectordb_file_path = "faiss_index"
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)
    retriever = vectordb.as_retriever(search_type="similarity", k=4)
    return retriever

def get_conversational_chain():
    prompt_template = """
    1. If the answer is found in the context:
    - Analyze the response text provided in the context.
    - Generate an answer in your own words, using the information from the context to craft a response that is clear and coherent.
    - Ensure that the response is not a direct copy from the response section, but rather a paraphrased version that conveys the same meaning.

    2. If the answer is not found in the context:
    - Provide a response based on your general knowledge or expertise in the subject matter.
    - Ensure that the answer is relevant and informative, even if it cannot be directly sourced from the context.
    
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain, prompt

def user_input(user_question, retriever, chain, prompt):
    retrieved_docs  = retriever.invoke(user_question)
    response = run_chain(chain, prompt, retrieved_docs, user_question)
    return response

def run_chain(chain, prompt, docs, user_question):
    response = chain.invoke(
        {"input_documents": docs, "question": user_question, "prompt": prompt},
        return_only_outputs=True)
    return response

def create_vectordb_retriever():
    data = load_data()
    retriever = create_embeddings(data)
    chain, prompt = get_conversational_chain()
    return retriever, chain, prompt

# Typing effect function
def type_text(text, speed=0.01):
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.markdown(f'<div class="typing-effect">{typed_text}</div>', unsafe_allow_html=True)
        time.sleep(speed)

def main():
    retriever, chain, prompt = create_vectordb_retriever()
    user_question = st.chat_input("Ask a Question")

    if user_question:
        response = user_input(user_question, retriever, chain, prompt)  
        # Save the conversation in session state
        st.session_state.chat_history.append({"user": user_question, "bot": response["output_text"]})
        
    # Display chat history
    if st.session_state['chat_history']:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            if chat['user']:
                st.markdown(
                    f'<div class="message-container user-container">'
                    f'<div class="message-bubble user-bubble">'
                    f'<span class="message-text">{chat["user"]}</span>'
                    f'</div>'
                    f'<span class="emoji">🙋‍♂️</span>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            if chat['bot']:
                st.markdown(
                    f'<div class="message-container bot-container">'
                    f'<span class="emoji">📡</span>'
                    f'<div class="message-bubble bot-bubble">'
                    f'<span class="message-text">{chat["bot"]}</span>'
                    f'</div>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            time.sleep(0.3)  
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
