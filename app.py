import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image


data = pd.read_csv('amazon_product.csv')
data = data.drop('id', axis=1)

stemmer = SnowballStemmer('english')

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    return stemmed
    #return " ".join(stemmed)

data['stemmed_tokens'] =  data.apply(lambda x: tokenize_stem(x['Title'] + " " + x['Description']), axis=1)

tfidf = TfidfVectorizer(tokenizer=tokenize_stem)
def cosine_sim(txt1, txt2):
    text1_concat = ' '.join(txt1)
    text2_concat = ' '.join(txt2)
    matrix = tfidf.fit_transform([text1_concat, text2_concat])

    return cosine_similarity(matrix)[0][1]

def search_product(query):
    stemmed_query = tokenize_stem(query)
    # calculating cosine similarity between query and stemmed token columns
    data['Similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(stemmed_query, x))
    res = data.assign(Similarity=data['Similarity'].apply(lambda x: np.max(x))).sort_values(by='Similarity', ascending=False).head(10)[['Title', 'Description', 'Category']]
    return res

img = Image.open('img.png')
st.image(img, width=600)
st.title("Search Engine and Amazon Product Recommendation")

query = st.text_input("Enter Product Name")
submit = st.button('Search')

if submit:
    result = search_product(query)
    st.write(result)