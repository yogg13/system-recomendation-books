import pickle
import streamlit as st
import numpy as np

st.header('System Recomendation Books')

# Load pickled data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_name.pkl', 'rb'))
rating_final = pickle.load(open('artifacts/rating_final.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

def fetch_poster(suggestion):
    book_names = []
    ids_index = []
    poster_urls = []

    for book_id in suggestion[0]:
        book_names.append(book_pivot.index[book_id])

    for name in book_names:
        ids = np.where(rating_final['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = rating_final.iloc[idx]['img_url']
        poster_urls.append(url)

    return poster_urls

def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_urls = fetch_poster(suggestion)
    
    for i in range(len(suggestion[0])):
        book_title = book_pivot.index[suggestion[0][i]]
        books_list.append(book_title)

    return books_list, poster_urls

selected_book = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_urls = recommend_book(selected_book)
    cols = st.columns(5)
    for i in range(1, 6):
        with cols[i-1]:
            st.text(recommended_books[i])
            st.image(poster_urls[i])
