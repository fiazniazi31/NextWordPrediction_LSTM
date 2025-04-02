import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = tf.keras.models.load_model('hamlet_model.h5')

# Load the tokenizer
with open('hamlet_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Get the max sequence length used during training
max_sequence_length = model.input_shape[1] + 1  # Adding 1 because input_shape excludes label

def predict_next_word(model, tokenizer, text, max_sequence_length):
    """Function to predict the next word based on input text."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) > max_sequence_length:
        token_list = token_list[-max_sequence_length:]
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit UI
st.title("Hamlet Text Predictor")
st.write("Enter a phrase from Shakespeare's Hamlet, and the model will predict the next word.")

# User input
input_text = st.text_input("Enter text:")

if st.button("Predict Next Word"):
    predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    if predicted_word:
        st.success(f"Predicted Next Word: {predicted_word}")
    else:
        st.error("Could not predict the next word. Try another input.")
