✈️ Airline Sentiment Analysis

📘 Overview
This project applies Natural Language Processing (NLP) and Machine Learning to classify airline tweets into Negative, Neutral, Positive sentiments.
It is deployed as an interactive Streamlit web app for real‑time sentiment prediction.

📂 Dataset
Source: Tweets.csv (Twitter airline sentiment dataset)
Records: ~14,640 tweets

Original columns:
tweet_id, airline_sentiment, airline_sentiment_confidence, negativereason, negativereason_confidence, airline, airline_sentiment_gold, name, negativereason_gold, retweet_count, text, tweet_coord, tweet_created, tweet_location, user_timezone

🧹 Preprocessing
Columns Dropped
tweet_id → unique identifier, not useful for prediction
airline_sentiment_confidence → confidence score, redundant with target label
negativereason_confidence → confidence score, not predictive
airline_sentiment_gold, negativereason_gold → gold labels, not needed for training
name, tweet_coord, tweet_created, tweet_location, user_timezone → metadata, not relevant to sentiment

Columns Kept
airline_sentiment → target variable (Negative, Neutral, Positive)
text → main feature (tweet content)
negativereason → categorical reason for negative sentiment (optional feature)
retweet_count → numeric feature, used for correlation analysis
sentiment_encoded → numeric encoding of target (Negative=0, Neutral=1, Positive=2)

⚙️ Project Flow
Data Loading → Import dataset into pandas.
Data Cleaning → Drop irrelevant columns, handle missing values.
Encoding → Map sentiment labels to numeric values.
Exploratory Data Analysis → Visualize distributions, correlations (heatmap of retweet_count vs sentiment_encoded).
Feature Extraction → Apply TF‑IDF vectorizer to text.
Model Training → Train Naive Bayes classifier.
Evaluation → Accuracy, confusion matrix.
Model Saving → Save trained model and vectorizer (.pkl).
Deployment → Build Streamlit app (app.py) for real‑time predictions.
GitHub + Streamlit Cloud → Upload repo, deploy app online.

Streamlit coloud Deployed link : 


📊 Results
Best model: Naive Bayes
Accuracy: ~76%
Strong recall for negative tweets (most common class).
Useful for monitoring customer feedback and improving airline services


