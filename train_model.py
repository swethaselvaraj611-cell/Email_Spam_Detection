import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle
# 1. Load dataset
data = pd.read_csv(
    "SMSSpamCollection.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# 2. Separate messages and labels
X = data["message"]
y = data["label"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Convert text into numerical features
vectorizer = TfidfVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# 5. Train the model
model = MultinomialNB()

model.fit(X_train_vectors, y_train)

# 6. Predict test data
y_pred = model.predict(X_test_vectors)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

print("\nConfusion Matrix:")
print(cm)

# 8. Save the model and vectorizer
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and Vectorizer saved successfully!")