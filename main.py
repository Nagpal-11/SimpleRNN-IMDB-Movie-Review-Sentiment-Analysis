import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# 1. Cache the word index download/loading
@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value : key for key , value in word_index.items()}
    return word_index, reverse_word_index

word_index, reverse_word_index = load_word_index()

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3 , '?') for i in encoded_review])

# 2. Cache the heavy model loading step
@st.cache_resource
def load_my_model():
    return load_model('simplernnimdb.h5')

model = load_my_model()

# Preprocessing the user input 
VOCAB_SIZE = 10000 

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [(word_index.get(word, 2) + 3) % VOCAB_SIZE for word in words]
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Streamlit app 
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("Please type something before clicking Classify!")
    else:
        preprocess_input = preprocess_text(user_input)
        prediction = model.predict(preprocess_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        
        st.subheader(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score : {prediction[0][0]:.4f}')
