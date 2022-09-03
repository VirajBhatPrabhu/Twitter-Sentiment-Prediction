import streamlit as st
import pickle
cv=pickle.load(open('countvectorizer.pkl','rb'))
clf=pickle.load(open('SA_Model.pkl','rb'))

page_bg_img = '''
<style>
.stApp {
  background-image: url("https://c4.wallpaperflare.com/wallpaper/154/47/615/twitter-social-networks-wallpaper-preview.jpg");
  background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Twitter Sentiment Prediction</h1>", unsafe_allow_html=True)


user = st.text_input(" ",placeholder='Enter a Tweet')
if st.button('Predict'):
    sample = user
    data = cv.transform([sample]).toarray()
    a = clf.predict(data)
    st.markdown(f"<h2 style='text-align: center;color:white;'>This is a {a[0]} tweet</h2>", unsafe_allow_html=True)