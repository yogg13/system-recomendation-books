from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html', book_names=book_names)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    book_name = data['book_name']
    recommended_books, poster_urls = recommend_book(book_name)
    return jsonify({
        'recommended_books': recommended_books,
        'poster_urls': poster_urls
    })

if __name__ == '__main__':
    app.run(debug=True)
