import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import random
import tweepy
import yaml

# Function to establish Twitter connection using version 2 method
def twitConnection(creds):
    consumer_key = creds['twitter']['consumer_key']
    consumer_secret = creds['twitter']['consumer_secret']
    access_token = creds['twitter']['access_token']
    access_secret = creds['twitter']['access_secret']
    
    client = tweepy.Client(
        consumer_key=consumer_key, consumer_secret=consumer_secret,
        access_token=access_token, access_token_secret=access_secret)
     
    return client

# Function to establish Twitter connection using version 1 method
def twitConnection_v1(creds):
    consumer_key = creds['twitter']['consumer_key']
    consumer_secret = creds['twitter']['consumer_secret']
    access_token = creds['twitter']['access_token']
    access_secret = creds['twitter']['access_secret']
    
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    return tweepy.API(auth)

# Load credentials from YAML file
creds = yaml.load(open('social_credentials.yml'), Loader=yaml.FullLoader)

# Establish Twitter connection
client = twitConnection(creds)  # or twitConnection_v1(creds) depending on the version of Tweepy you're using

# Load your dataset
df = pd.read_csv('Bullytweet.csv')

# Ensure that the 'TwitterID' column is the index
df.set_index('TwitterID', inplace=True)

# Drop rows with missing values in the 'Tweet' column
df.dropna(subset=['Tweet'], inplace=True)

# Apply sentiment analysis using TextBlob
df['Sentiment'] = df['Tweet'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Tweet'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.2, random_state=42)

# Train the classifier (Multinomial Naive Bayes)
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Predict the label for a new tweet
new_tweet = "Annamalai is good"
new_tweet_vectorized = vectorizer.transform([new_tweet])
new_tweet_label = clf.predict(new_tweet_vectorized)[0]
print(f"Prediction for the new tweet: {new_tweet_label}")
