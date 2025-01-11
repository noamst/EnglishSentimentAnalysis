import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import streamlit as st

# Load model and tokenizer
model = load_model('LSTM_sentiment_analysis.h5')

import json

# Load the word index
with open('word_index.json', 'r') as f:
    word_index = json.load(f)

def encode_sentence(sentence, word_index, oov_token_index=1):
    """
    Encode a sentence using the word_index dictionary.
    
    Args:
        sentence (str): The input sentence to encode.
        word_index (dict): The word-to-index mapping.
        oov_token_index (int): Index for out-of-vocabulary (OOV) words.
    
    Returns:
        list: List of integers representing the encoded sentence.
    """
    # Tokenize the sentence into words
    words = sentence.split()  # Simple whitespace-based tokenization
    encoded = [word_index.get(word, oov_token_index) for word in words]
    return encoded

def sentiment_predict(model,sentence):
    categories_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Tokenize and preprocess input
    myEncoder = encode_sentence(sentence,word_index)
    print("Encoded Input:", myEncoder)
    padded_input = pad_sequences([myEncoder], padding='pre', maxlen=128)
    print("Padded Input:", padded_input)

    # Make prediction
    prediction = model.predict(padded_input)
    print("Prediction:", prediction)

    # Determine sentiment and score
    sentiment = categories_map[np.argmax(prediction)]
    score = np.max(prediction)
    return sentiment, score

# Streamlit app
st.title('Sentiment Analysis')
st.write('Enter a sentence to classify it as positive, neutral, or negative.')

# User input
user_input = st.text_area('Sentence')

if st.button('Classify'):
    if user_input.strip():
        sentiment, score = sentiment_predict(model,user_input)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {score:.2f}')
    else:
        st.write("Please enter a valid sentence.")
else:
    st.write('Awaiting input...')
