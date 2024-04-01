import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.metrics import accuracy_score

# Read the CSV file
df = pd.read_csv('Twitter_Data.csv', encoding='ISO-8859-1')

print(df.head(5))

print(df.isnull().sum())

# Drop rows with missing values in the 'text' and 'gender' columns
df.dropna(subset=['text', 'gender'], inplace=True)

print(df.isnull().sum())

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Apply preprocessing to 'text' column
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Initialize CountVectorizer
count_vectorizer = CountVectorizer(max_features=5000)

# Fit and transform the text data
count_matrix = count_vectorizer.fit_transform(df['cleaned_text'])

# Split the data into features (X) and target labels (y)
X = count_matrix
y = df['gender']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the CountVectorizer object
with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(count_vectorizer, vectorizer_file)

# Save the trained classifier
with open('nb_classifier.pkl', 'wb') as classifier_file: 
    pickle.dump(nb_classifier, classifier_file)
