import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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

# Drop duplicate Twitter IDs
df.drop_duplicates(inplace=True)

# Apply sentiment analysis using TextBlob
df['Sentiment'] = df['Tweet'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Text preprocessing and model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Label'], test_size=0.2, random_state=42)

# Train the classifier and evaluate using cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# Train the classifier on the full training set
pipeline.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)
print("Test set performance:")
print(classification_report(y_test, y_pred))

# Predict a new tweet
new_tweet = "K. Annamalai's dedication to serving the people of Tamil Nadu is truly commendable. #Respect"
predicted_label = pipeline.predict([new_tweet])[0]
print("\nPrediction for the new tweet:", predicted_label)
