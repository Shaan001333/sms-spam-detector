import nltk
nltk.download('punkt', quiet=True)  # Fixes tokenizer error
nltk.download('stopwords', quiet=True)  # For spam detection
nltk.data.path.append("nltk_data")
import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
tfidf = pickle.load(open("vectorizer.pkl" , "rb"))
model = pickle.load(open("model.pkl" , "rb"))

st.title("Email / SMS spam Detector")
input_sms = st.text_area ("Enter the message")

if st.button("Predict"):
    transform_sms = transform_text(input_sms)
    # st.write("Transformed Text:", transform_sms)

    vector_input = tfidf.transform([transform_sms])
    # st.write("Non-zero TF-IDF Elements:", vector_input.nnz)

    result = model.predict(vector_input)[0]
    # st.write("Prediction Result:", result)

    if result == 1:
        st.header("⚠️ This message is SPAM!")
    else:
        st.header("✅ Cool! This message is NOT spam.")

