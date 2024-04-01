import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
     count_vectorizer = pickle.load(vectorizer_file)
with open('nb_classifier.pkl', 'rb') as classifier_file:
     nb_classifier = pickle.load(classifier_file)

def preprocess_text(text):
    text= text.lower()
    text= re.sub(r'http\S+','',text)
    text= re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text= re.sub (r'#','', text)
    text= re.sub(r'[a-zA-Z\s]','', text)
    return text

sentiment_mapping = {
    -1 :'negative',
    0: 'neutral',
    1: 'positive'
    }

def main():
    coll, col2, col3, co14= st.columns([1,1,3,1])
    with col3:
        st.image("twitter.png", width=100)
    st.title("Twitter Sentiment Classifier")
    st.write("Enter a twitter test below:")
    input_text = st.text_area("Input Text: ","")
    if st.button("Predict"):
       cleaned_text= preprocess_text(input_text)
       vectorizer_text = count_vectorizer.transform([cleaned_text])
       sentiment_prediction =nb_classifier.predict(vectorizer_text)[0]

       predicted_sentiment = sentiment_mapping.get(sentiment_prediction, 'Unknown Sentiment')