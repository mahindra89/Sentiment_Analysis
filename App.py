import os
import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
# Define the path to your tokenizer and model
path = r'C:\Users\Tarek Hesham\Sent_model'

#setting configuration
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
config.hidden_dropout_prob = 0.2  
config.attention_probs_dropout_prob = 0.2 

# Load tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(path, 'Tokenizer'))

# Initialize a model with the same architecture (use the same version of BERT you trained with)
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config = config)

# Load weights from your custom .h5 file
model_weights_path = os.path.join(path, 'Model', 'tf_model_new.h5')  # replace with your actual file name
bert_model.load_weights(model_weights_path)

#label encoding
label = {0:"neutral", 1:"positive", 2:"negative"}

def Get_sentiment(Review, Tokenizer=bert_tokenizer, Model=bert_model):
    # Convert Review to a list if it's not already a list
    if not isinstance(Review, list):
        Review = [Review]
 
    Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
                                                                             padding=True,
                                                                             truncation=True,
                                                                             max_length=128,
                                                                             return_tensors='tf').values()
    prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])
 
    # Use argmax along the appropriate axis to get the predicted labels
    pred_labels = tf.argmax(prediction.logits, axis=1)
 
    # Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
    pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
    return pred_labels

st.title("Sentiment Analysis App")

# User input area
review = st.text_area("Enter a review to analyze")

# Predict button
if st.button("Predict Sentiment"):
    if review:
        result = Get_sentiment(review)
        st.write(f"Predicted Sentiment: {result}")
    else:
        st.write("Please enter some text.")