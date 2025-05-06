import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer





#load resources

lemmatizer = WordNetLemmatizer()
stop_words =  set(stopwords.words('english'))
model = joblib.load('nb_model.pkl')

#page configuration
st.set_page_config(page_title='news categorization',page_icon="<3",layout='centered')

st.title("News categorization using nltk !")
st.markdown("enter a news headline or article below. ")
st.markdown('___')
input_text = st.text_area("enter news text: ",height=200)


# text preprocessing function 

def preprocess_text(text):
    text= text.lower()

    #remove special character and punctuation 

    text = re.sub(r'[^a-zA-Z\s]', '' ,text)

    # tokenization

    words = word_tokenize(text)

    # remove stopwords and lemantize
    words = [lemmatizer.lemmatize(wor) for wor in words if wor not in stop_words]
    

    #join back into a string 

    return " " .join(words)



# prediction function 

def Predict_category(text):
    prcessed_text = preprocess_text(text)
    prediction = model.predict([prcessed_text])
    return prediction[0]


#prediction display 

if st.button('Predict Category'):
    if input_text.strip():
        category = Predict_category(input_text)
        st.success(f'**Predicted category: ** {category}')
    else:
        st.warning("please enter some text")
        

#footer 

st.markdown("---")
st.markdown('<p style="text-align:center;">Built by frail2004 </p>',unsafe_allow_html=True)

