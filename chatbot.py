import streamlit as st
from pathlib import Path
import tensorflow as tf
import pickle

# --- Page Config ---
st.set_page_config(page_title="Benjamin v1", layout="wide")
st.title("Benjamin")

# --- CSS Styling ---
st.markdown("""
    <style>
    .message-bubble {
        padding: 10px;
        border-radius: 10px;
        margin: 10px;
        max-width: 70%;
        width: fit-content;
    }
    .user-message {
        background-color: #FFEFDF;
        text-align: left;
        margin-left: auto;
    }
    .chatbot-message {
        background-color: #FF9900;
        text-align: left;
        margin-right: auto;
        color: white;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        width: 100%;
    }
    .message-row {
        display: flex;
        align-items: flex-start;
    }
    .message-row.user {
        justify-content: flex-end;
    }
    .message-row.bot {
        justify-content: flex-start;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Chatbot Initialization ---

# Load the Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/Benjamin_v1.keras')
    return model

model = load_model()

# Load the tokenizer (pickle file)
@st.cache_resource
def load_tokenizer():
    with open('model/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

tokenizer = load_tokenizer()

def tokenize_input(user_input):
    # Convert user input to a sequence of integers using your tokenizer
    sequence = tokenizer.texts_to_sequences([user_input])
    # Pad the sequence if necessary (adjust maxlen as needed)
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=50, padding='post')
    return padded_sequence

def decode_response(predicted_seq):
    # Assuming you have a dictionary to map indices to words
    response = ' '.join([tokenizer.index_word.get(idx, '') for idx in predicted_seq[0] if idx > 0])
    return response


# --- Initialize or Load Chat History from session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display Chat History ---
for user_msg, chatbot_msg in st.session_state.chat_history:
    # User Message (Right-Aligned)
    with st.container():
        st.markdown(f"""
        <div class="message-row user">
            <div class="message-bubble user-message">{user_msg}</div>
            <img src="https://cdn-icons-png.flaticon.com/512/1946/1946429.png" class="avatar"/>
        </div>
        """, unsafe_allow_html=True)

    # Bot Message (Left-Aligned with Avatar on the left)
    with st.container():
        st.markdown(f"""
        <div class="message-row bot">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" class="avatar"/>
            <div class="message-bubble chatbot-message">{chatbot_msg}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Input Field ---
user_input = st.chat_input("Ask me anything")
if user_input:
    # Display user message
    with st.container():
        st.markdown(f"""
        <div class="message-row user">
            <div class="message-bubble user-message">{user_input}</div>
            <img src="https://cdn-icons-png.flaticon.com/512/1946/1946429.png" class="avatar"/>
        </div>
        """, unsafe_allow_html=True)

    # Simulate chatbot response
        tokenized_input = tokenize_input(user_input)
        predicted_seq = model.predict(tokenized_input)
        response = decode_response(predicted_seq)


    if response:
        # Display chatbot response
        with st.container():
            st.markdown(f"""
            <div class="message-row bot">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" class="avatar"/>
                <div class="message-bubble chatbot-message">{response}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Store the current chat in session state for future reference
        st.session_state.chat_history.append((user_input, response))
