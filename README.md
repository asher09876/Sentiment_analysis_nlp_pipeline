# Sentiment_analysis_nlp_pipeline

This project is an end to end machine learning pipeline that works on analyzing news articles related to Artificial Intelligence. The goal was to not only train a model but also to practice the entire machine learning lifecycle. The pipeline includes data ingestion, validation, preprocessing, training, evaluation, and deployment with an API.

## Features of the Project

Data ingestion from live news using the NewsAPI
Data validation that cleans raw data and removes null values and duplicates
Text preprocessing including lowercasing, stopword removal, lemmatization and sentiment labeling
Feature engineering using TF IDF vectors
Model training with Logistic Regression as a baseline and a custom neural network using Keras
Model evaluation with accuracy, precision, recall, f1 score and confusion matrix
Model serving through FastAPI with a predict endpoint

### Step 1 Clone the repository
git clone https://github.com/your-username/nlp-news-sentiment.git
cd nlp-news-sentiment
### Step 2 Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac or Linux
### Step 3 Install dependencies
pip install -r requirements.txt
### Step 4 Set your API key
Create a free account at NewsAPI and copy the API key.
setx NEWS_API_KEY "your_api_key_here"
### 
