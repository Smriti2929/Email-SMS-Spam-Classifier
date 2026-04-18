import pickle
import streamlit as st
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# download
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

# Load saved model + vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = nltk.word_tokenize(text)

    y = []
    stop_words = set(stopwords.words('english'))

    important_words = {'off', 'free', 'win', 'call', 'now', 'buy'}

    for i in text:
        if (i not in stop_words) or (i in important_words):
            y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
st.title("📩Email/ SMS Spam Classifier")

input_sms = st.text_area("Enter your message")

if st.button('Predict'):

    if input_sms.strip() != "":

        # preprocess
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = tfidf.transform([transformed_sms])

        # predict
        result = model.predict(vector_input)[0]

        #display
        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter a message")

# Demo examples
st.markdown("### Try examples:")
st.write("WIN FREE CASH NOW!!!")
st.write("Congratulations! Claim your prize now")
st.write("Let's meet for lunch tomorrow")

# 1. preprocess
# 2. vectorize
# 3. predict
# 4. display

