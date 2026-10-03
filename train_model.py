import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# ==============================
# 1. Create Dataset
# ==============================

spam_messages = [
    "Congratulations! You have won a cash prize. Claim now.",
    "URGENT! You have been selected for a special reward. Call now.",
    "Win a free gift card today. Click to claim your prize.",
    "You are a lucky winner. Send your details to receive the reward.",
    "Exclusive offer! Get a huge discount today. Limited time only.",
    "Congratulations, you won a free vacation. Claim your ticket now.",
    "Your account has won a bonus reward. Verify now.",
    "FREE recharge available today. Claim your offer immediately.",
    "You have been chosen for a special cash reward.",
    "Limited time offer! Get your bonus before it expires."
]

ham_messages = [
    "Hi, are we still meeting after class today?",
    "Please send me the notes when you get time.",
    "Your appointment is confirmed for tomorrow at 10 AM.",
    "Can you call me when you reach home?",
    "The project meeting has been moved to Monday.",
    "Don't forget to bring your notebook to class.",
    "I will send the assignment file tonight.",
    "Lunch is ready. Come downstairs when you can.",
    "Thanks for helping me with the project.",
    "The bus will arrive at the usual time."
]


rows = []

# 100 spam messages
for i in range(100):
    rows.append({
        "id": i + 1,
        "message": spam_messages[i % len(spam_messages)],
        "label": "spam"
    })

# 100 normal messages
for i in range(100):
    rows.append({
        "id": i + 101,
        "message": ham_messages[i % len(ham_messages)],
        "label": "ham"
    })


data = pd.DataFrame(rows)

# Save dataset
data.to_csv(
    "ai_spam_detector_200_rows.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Dataset created successfully!")
print("Total rows:", len(data))
print("Spam:", (data["label"] == "spam").sum())
print("Ham:", (data["label"] == "ham").sum())


# ==============================
# 2. Prepare Data
# ==============================

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 3. Create AI Model
# ==============================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])


# ==============================
# 4. Train Model
# ==============================

model.fit(X_train, y_train)


# ==============================
# 5. Save Model
# ==============================

joblib.dump(model, "spam_detector_model.pkl")

print()
print("Model successfully trained!")
print("Model saved as spam_detector_model.pkl")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))