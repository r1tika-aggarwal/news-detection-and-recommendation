# news-detection-and-recommendation
This project is a Flask-based web application that combines AI/ML techniques for fake news detection and personalized news recommendation.
The system analyzes news articles to determine their authenticity and provides users with relevant news recommendations based on their preferences.

Features:
1. Fake News Detection: Determine whether a news article is real or fake.
2. News Recommendation: Get personalized news recommendations.

Prerequisites:
Python 3.x,
Flask,
Scikit-learn,
Pandas,
Numpy.

Installation: 
1. Clone the repository
2. Create a virtual environment:
>> python -m venv venv
3. Activate the virtual environment:
>> venv\Scripts\activate
4. Install the required packages:
>> pip install -r requirements.txt

Dataset: 
To run the project, you need to download the dataset from the following link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Usage:
1. Run FNDusingtfidf: First, you must run this file to generate the necessary pickle files (.pkl), which contain the trained models and vectorizers.
2. Run app.py: After running FNDusingtfidf, you can start the Flask application using app.py
3. Access the Application: Open your web browser and navigate to http://127.0.0.1:5000/ to interact with the application.

Contributing:
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
