import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import re, string
from tensorflow.keras.utils import register_keras_serializable
import pickle
import pandas as pd

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
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    # Keep < and >, remove all other punctuation
    return tf.strings.regex_replace(lowercase, r"[^\w\s<>]", "")

# Encoder Layer
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

# Decoder Layer
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

# Load Model
model = tf.keras.models.load_model("model/benjamin_v1_5.keras", compile=False)

# Load vectorizer
with open('model/vectorizer_5.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load data and format
pairs_df = pd.read_csv("model/train_dialog.csv")
prompt_response_pairs = list(zip(pairs_df['prompt'], pairs_df['response']))

# Pad punctuation
def pad_punctuation(s):
    pattern = r"([{}])".format(re.escape(string.punctuation))
    s = re.sub(pattern, r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s.strip()

# Prepare data
text_data = [(pad_punctuation(p), pad_punctuation(r)) for p, r in prompt_response_pairs]
formatted_data = [("<start> " + p + " <end>", "<start> " + r + " <end>") for p, r in text_data]
inputs = [inp for inp, _ in formatted_data] 
targets = [tgt for _, tgt in formatted_data]

combined_inputs = inputs + targets
vectorizer.adapt(combined_inputs)
vocab = vectorizer.get_vocabulary()
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

# Vectorize input
def vectorize_input(input_text):
    input_tensor = tf.convert_to_tensor([input_text])
    vectorized_input = vectorizer(input_tensor)
    return vectorized_input

# Top-p Sampling
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

# Initialize Encoder and Decoder
encoder = Encoder(len(vocab), EMBED_DIM, LATENT_DIM)
decoder = Decoder(len(vocab), EMBED_DIM, LATENT_DIM)

# Generate response
def generate_response(input_text, p=PROB, temperature=TEMPERATURE):
    input_text = pad_punctuation(input_text)  # Format the input
    input_text = "<start> " + input_text + " <end>"  # Add start/end tokens
    input_seq = vectorizer([input_text])  # Tokenize and vectorize the input
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
st.title("🤖 Benjamin v1 Chatbot")
user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    response = generate_response(user_input)
    st.text_area("Bot:", value=response, height=100)
