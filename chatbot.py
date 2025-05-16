import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np
import re, string
from tensorflow.keras.utils import register_keras_serializable
import pickle
import pandas as pd


import base64

def get_svg_base64(path):
    with open(path, "rb") as f:
        svg_bytes = f.read()
    return base64.b64encode(svg_bytes).decode("utf-8")
benjamin_svg_base64 = get_svg_base64("static/benjamin_v1.svg")


# --- Page Config ---
st.set_page_config(page_title="ChatBot", layout="wide")
st.title(" Benjamin v1 ")
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











# Constants
OUTPUT_SEQ_LENGTH = 50
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EMBED_DIM = 256
LATENT_DIM = 512
EPOCHS = 1
TEMPERATURE = 0.7
PROB = 0.9
@register_keras_serializable()
class StartTokenLayer(tf.keras.layers.Layer):
    def __init__(self, start_token_idx, **kwargs):
        super().__init__(**kwargs)
        self.start_token_idx = start_token_idx

    def call(self, inputs):
        return tf.fill([tf.shape(inputs)[0], 1], self.start_token_idx)
# Custom Standardization function for vectorizer
@register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    # Keep < and >, remove all other punctuation
    return tf.strings.regex_replace(lowercase, r"[^\w\s<>]", "")

# Encoder Layer (same as before)
@register_keras_serializable()
class Encoder(layers.Layer):
    def __init__(self, vocab_size, embed_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.lstm = layers.LSTM(latent_dim, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return state_h, state_c

# Decoder Layer (same as before)
@register_keras_serializable()
class Decoder(layers.Layer):
    def __init__(self, vocab_size, embed_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size, activation="softmax")

    def call(self, x, state_h, state_c):
        x = self.embedding(x)
        x, _, _ = self.lstm(x, initial_state=[state_h, state_c])
        return self.dense(x)


# Pad punctuation function (same as before)
def pad_punctuation(s):
    pattern = r"([{}])".format(re.escape(string.punctuation))
    s = re.sub(pattern, r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s.strip()

# Load both parts
loaded_seq2seq_model = models.load_model("model/benjamin_v1.keras")
vectorizer = models.load_model("model/vectorizer_v1.keras")

# Extract encoder and decoder from the loaded model
encoder = loaded_seq2seq_model.get_layer("encoder")  # Replace with your actual encoder layer name
decoder = loaded_seq2seq_model.get_layer("decoder") 


text_vectorization_layer = vectorizer.layers[0]

# Now you can call get_vocabulary() on the actual layer
vocab = text_vectorization_layer.get_vocabulary()
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}


def vectorize_input(input_text):
    input_tensor = tf.convert_to_tensor([input_text])
    print("tensor:", input_tensor)
    vectorized_input = vectorizer(input_tensor)
    print("vector: ", vectorized_input)
    return vectorized_input

# Top-p Sampling function (same as before)
def top_p_sampling(probabilities, p=PROB, temperature=1.0):
    if temperature != 1.0:
        logits = np.log(probabilities + 1e-10) / temperature
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)

    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cumulative_probs, p)
    top_p_indices = sorted_indices[:cutoff + 1]
    top_p_probs = probabilities[top_p_indices]
    top_p_probs /= np.sum(top_p_probs)

    sampled_index = np.random.choice(top_p_indices, p=top_p_probs)
    return sampled_index



# Generate response function (same as before)
def generate_response(input_text, p=PROB, temperature=TEMPERATURE):
    input_text = pad_punctuation(input_text)  # Format the input
    print("input pad: ", input_text)
    input_text = "<start> " + input_text + " <end>"  # Add start/end tokens
    print("intput token", input_text)
    input_seq = vectorizer(tf.convert_to_tensor([input_text]))  # Tokenize and vectorize the input
    state_h, state_c = encoder(input_seq)  # Get the states from the encoder

    target_seq = tf.constant([[word_to_index["<start>"]]])  # Initialize the target sequence as <start>
    output_text = []

    for _ in range(50):
        # Ensure the decoder is returning valid logits
        pred = decoder(target_seq, state_h, state_c)
        
        if pred is None:  # Check if decoder output is None
            raise ValueError("Decoder returned None. Check decoder layer implementation.")
        
        pred_probs = pred[0, -1, :].numpy()  # Get probabilities for the last word
        sampled_token_index = top_p_sampling(pred_probs, p=p, temperature=temperature)  # Sample from top-p distribution
        sampled_word = index_to_word[sampled_token_index]  # Get the word from the sampled index

        if sampled_word == "<end>":  # End the generation if <end> token is sampled
            break

        output_text.append(sampled_word)  # Append the predicted word to the output list
        target_seq = tf.concat([target_seq, [[sampled_token_index]]], axis=-1)  # Update target sequence

        # Prevent repeating the same word three times in a row
        if len(output_text) > 3 and output_text[-1] == output_text[-2] == output_text[-3]:
            break

    return " ".join(output_text)  # Return the generated sentence

# Streamlit UI







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
            <img src="data:image/svg+xml;base64,{benjamin_svg_base64}" class="avatar"/>
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
        response = generate_response(user_input)
        
        if response:
            # Display chatbot response
            with st.container():
                st.markdown(f"""
                <div class="message-row bot">
                    <img src="data:image/svg+xml;base64,{benjamin_svg_base64}" class="avatar"/>
                    <div class="message-bubble chatbot-message">{response}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Store the current chat in session state for future reference
            st.session_state.chat_history.append((user_input, response))




