# app.py
from flask import Flask, render_template, request
import joblib
import random

app = Flask(__name__)

# Load the trained model and vectorizer (use your actual model and vectorizer)
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Home route to display the website
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submissions (message classification)
@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the user
    user_message = request.form['message']

    # Vectorize the message and predict
    message_vectorized = vectorizer.transform([user_message])
    prediction = model.predict(message_vectorized)

    # Classify the message and return result
    result = 'Suspicious' if prediction == 1 else 'Not Suspicious'
    return render_template('index.html', message=user_message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
