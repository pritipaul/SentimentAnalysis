# import streamlit as st
# import tensorflow as tf
# from keras.models import load_model
# import numpy as np
# from keras.preprocessing.text import Tokenizer


# st.title("Streamlit App ")

# model = load_model('Model-GRU.h5')
# user_input = st.text_area("Enter a product review:")


# if user_input:
#     if st.button("Predict"):
#         prediction = model.predict([user_input])
#         st.write(f"Predicted Sentiment: {prediction}")
#         st.write("Please enter a review to enable the 'Predict' button.")

# else:
#     st.write("Please enter a review to enable the 'Predict' button.")

import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('Model-GRU.h5')

# Initialize the tokenizer (ensure you load it with the same parameters used during training)
tokenizer = Tokenizer(num_words=5000)  # Adjust num_words as per your model's requirements

# Title of the app
st.title("Streamlit App")

# User input
user_input = st.text_area("Enter a product review:")

if user_input:
    if st.button("Predict"):
        # Tokenize and pad the input
        sequences = tokenizer.texts_to_sequences([user_input])
        padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as per your model's requirements
        
        # Make prediction
        prediction = model.predict(padded_sequences)
        
        # Output the prediction
        st.write(f"Predicted Sentiment: {prediction[0]}")  # Assuming prediction returns probabilities

else:
    st.write("Please enter a review to enable the 'Predict' button.")
