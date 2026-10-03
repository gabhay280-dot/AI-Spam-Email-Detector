# AI Based Spam Email/SMS Detector

An AI and Machine Learning based web application that detects whether a given Email or SMS message is Spam or Not Spam.

## Project Overview

The AI Based Spam Email/SMS Detector analyzes text messages and classifies them into two categories:

- Spam
- Ham (Not Spam)

The system uses Natural Language Processing (NLP) and Machine Learning techniques to analyze the text.

## Technologies Used

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes

### Other Tools
- Pandas
- Joblib
- CSV Dataset

## Machine Learning Workflow

The system follows these steps:

1. User enters an Email/SMS message.
2. Frontend sends the message to the Flask backend.
3. The trained ML model receives the message.
4. TF-IDF converts the text into numerical features.
5. Multinomial Naive Bayes classifies the message.
6. The system returns Spam or Not Spam.
7. A confidence score is displayed on the dashboard.
8. The prediction is stored in recent history.

## Project Structure

```text
AI-Spam-Detector/
│
├── app.py
├── train_model.py
├── requirements.txt
├── spam_detector_model.pkl
├── ai_spam_detector_200_rows.csv
│
└── templates/
    └── index.html