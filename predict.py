import streamlit as st
from PIL import Image, ImageDraw
import json
from streamlit_lottie import st_lottie
import pandas  as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
import chardet
from joblib import dump
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

spam = pd.read_csv('mail_data.csv')
print(spam)

#labelling spam as 0 and ham as 1
spam.loc[spam['Category'] == 'spam','Category'] = 0
spam.loc[spam['Category'] == 'ham','Category'] = 1
print(spam)

#seperating as message and spam or ham
X = spam['Message']
Y = spam['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
#random state to split the data equally each time

model = load('spam_email_prediction.joblib')

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="Hello"
)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_coding = load_lottiefile("C:\\Mini-Project\\ghello.json")
st_lottie(
    lottie_coding,
    height=400,
)

st.sidebar.header('SPAM EMAIL DETECTOR')
st.sidebar.text('Do not let spam get in the way - try our email detector.')
image = Image.open('spam.png')
st.image(image)
st.header("SPAM EMAIL DETECTOR")
st.text("Spam email refers to unsolicited and unwanted messages sent to a large number of people, ")
st.text("often for malicious or fraudulent purposes. These messages are usually sent in bulk and ")
st.text("contain irrelevant or inappropriate content that is intended to deceive the recipient into ")
st.text("contain irrelevant or inappropriate content that is intended to deceive the recipient into ")
st.text("emails can also be used to gather personal information or to scam individuals out of their ")
st.text("money. They are a significant problem for individuals, businesses, and organizations,")
st.text("as they can be time-consuming to deal with and can pose a serious threat to cyber-")
st.text("security.To avoid spam email, it is important to be cautious when providing")
st.text("personal information online, to use spam filters or email blockers, and to avoid")
st.text("clicking on links or downloading attachments from unknown senders.")

st.subheader("Enter the email in the textbox below and press enter to check whether the email is authentic email or a spam email.")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_coding = load_lottiefile("C:\\Mini-Project\\arrow_down.json")
st_lottie(
    lottie_coding,
    height=100,
)

features = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = features.fit_transform(X_train)
X_test_features = features.transform(X_test)

title = st.text_input("Enter the Email text here")
if title:
    input_features = features.transform([title])
    
    prediction = model.predict(input_features)
    if prediction[0]==1:
        st.write("The mail is authentic")


    if prediction[0]==0:
        st.write("ALERT!!!It is a spam mail")
