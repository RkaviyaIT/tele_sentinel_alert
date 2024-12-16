# training.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import nltk
nltk.download('stopwords')

# Load and preprocess data
data = pd.read_csv('data/manual_labeled_messages.csv')

# Clean text (lowercase, remove stopwords)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
data['message'] = data['message'].str.lower()
data['message'] = data['message'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Vectorize the text (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['message'])
y = data['label']  # Labels for scam (1) or not (0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and vectorizer
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
