import webbrowser
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import random
import tweepy
import yaml

app = Flask(__name__)

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

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report', methods=['POST'])
def report():
    tweet_index = int(request.form['tweet_index']) - 1
    bully_indices = df[df['Label'] == 'Bully'].index.tolist()
    selected_tweet = df.loc[bully_indices[tweet_index], 'Tweet']
    user_id = bully_indices[tweet_index]

    # Construct the tweet message for reporting
    politician_handle = "@annamalai007"
    report_message = f"@Twitter, this user ''@{user_id}'' bully tweeted as ''{selected_tweet}'' and seems as bullying content about the politician {politician_handle}, please take action!"

    # Post the tweet
    try:
        response = client.create_tweet(text=report_message)
        return render_template('success.html')
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    # Open the webpage automatically
    webbrowser.open_new_tab('http://127.0.0.1:5000/')
    # Run the Flask app
    app.run(debug=True)