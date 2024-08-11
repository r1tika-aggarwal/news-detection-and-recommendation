from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import nltk
import joblib
import logging

# Ensure necessary nltk data is downloaded
nltk.download('stopwords')

app = Flask(__name__)

# Load the fake news model, vectorizer, and PCA
with open('best_rf_classifier_new.pkl', 'rb') as model_file:
    fake_news_model = joblib.load(model_file)

with open('tfidf_vectorizer_new.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = joblib.load(vectorizer_file)

with open('pca_new.pkl', 'rb') as pca_file:
    pca = joblib.load(pca_file)

# Load data and preprocess
data = pd.read_csv("news_recommendation.csv")
data = data.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "title_summary"], errors='ignore')
data = data.dropna()
data = data.drop_duplicates(subset=None, keep='first', inplace=False)
data.insert(0, 'id', range(0, data.shape[0]))

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    texts = [w for w in text if w.isalpha()]
    texts = " ".join(texts)
    return texts

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Preprocess News Data
data['cleaned_desc'] = data['text'].apply(make_lower_case)
data['cleaned_desc'] = data.cleaned_desc.apply(remove_stop_words)
data['cleaned_desc'] = data.cleaned_desc.apply(remove_punctuation)
data['cleaned_desc'] = data.cleaned_desc.apply(remove_html)

# TF-IDF Vectorization
tf = TfidfVectorizer(analyzer='word', stop_words='english', use_idf=True, ngram_range=(1,1))
tfidf_matrix = tf.fit_transform(data['cleaned_desc'])

# Preprocess Text for Fake News Detection
def preprocess_text(text):
    text = make_lower_case(text)
    text = remove_punctuation(text)
    text = remove_html(text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')

@app.route('/check_fake_news', methods=['POST'])
def check_fake_news():
    data = request.json
    news_content = data.get('news_content')

    if news_content:
        preprocessed_content = preprocess_text(news_content)
        tfidf_vector = tfidf_vectorizer.transform([preprocessed_content])
        
        # Apply PCA transformation
        pca_vector = pca.transform(tfidf_vector.toarray())
        
        # Make prediction using the reduced features
        prediction = fake_news_model.predict(pca_vector)[0]
        is_real_news = bool(prediction)  # Ensure it is explicitly a boolean
        
        # Log the output
        logging.info(f"Prediction: {prediction}, is_real_news: {is_real_news}")
        
        return jsonify({'is_real_news': is_real_news})
    else:
        return jsonify({'error': 'Invalid input'})

# Function for recommendations based on keywords with simplified TF-IDF
def recommend_by_keywords(keywords, no_of_news_article):
    # Create a temporary DataFrame for the keyword-based search
    cleaned_keyword = remove_punctuation(remove_stop_words(make_lower_case(keywords)))
    keyword_df = pd.DataFrame({
        'title': [keywords],
        'cleaned_desc': [cleaned_keyword]
    })

    # Vectorize the keywords
    keyword_tfidf_matrix = tf.transform(keyword_df['cleaned_desc'])

    # Calculate cosine similarities between the keyword-based TF-IDF and the existing articles
    keyword_cosine_similarities = linear_kernel(keyword_tfidf_matrix, tfidf_matrix)

    # Check if all similarity scores are zero
    if not keyword_cosine_similarities.any():
        return []

    # Get similarity scores
    similarity_score = list(enumerate(keyword_cosine_similarities[0]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[:no_of_news_article]

    recommendations = []
    news_indices = [i[0] for i in similarity_score]
    for i in range(len(news_indices)):
        recommendations.append({
            'date': str(data['date'].iloc[news_indices[i]]),
            'title': data['title'].iloc[news_indices[i]],
            'link': data['link'].iloc[news_indices[i]]
        })
    return recommendations

@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
    recommendations = None
    keywords = None
    if request.method == 'POST':
        keywords = request.form['keywords']
        num_articles = int(request.form['num_articles'])
        recommendations = recommend_by_keywords(keywords, num_articles)
    return render_template('recommender.html', recommendations=recommendations, keywords=keywords)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)