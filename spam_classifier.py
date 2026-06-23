# ===============================
# Simple Spam Classifier
# User Input with Exit Condition
# Using Naive Bayes
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)


# Vectorization
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# Model accuracy
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


# ---------- USER INPUT LOOP ----------
print("\nType a message to check spam.")
print("Type 'exit' to stop the program.\n")

while True:
    user_msg = input("Enter message: ")

    if user_msg.lower() == "exit":
        print("Program exited 👋")
        break

    user_msg_vec = vectorizer.transform([user_msg])
    result = model.predict(user_msg_vec)

    if result[0] == 1:
        print("Result: SPAM 🚫\n")
    else:
        print("Result: NOT SPAM ✅\n")
