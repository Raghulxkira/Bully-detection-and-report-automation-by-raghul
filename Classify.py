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

# Drop duplicate Twitter IDs
df.drop_duplicates(inplace=True)

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

# Randomly select 5 tweets (3 bully and 2 non-bully)
bully_indices = df[df['Label'] == 'Bully'].sample(n=3).index.tolist()
non_bully_indices = df[df['Label'] == 'Non-Bully'].sample(n=2).index.tolist()
selected_indices = bully_indices + non_bully_indices
random.shuffle(selected_indices)

# Display randomly selected tweets with predicted labels and Twitter IDs
print("\nRandomly selected tweets with predicted labels and Twitter IDs:")
print("-" * 100)
for i, index in enumerate(selected_indices, start=1):
    tweet_series = df.loc[index, 'Tweet']
    # Check if tweet_series is a pandas Series
    if isinstance(tweet_series, pd.Series):
        tweet = tweet_series.iloc[0]  # Access the first element if it's a Series
    else:
        tweet = tweet_series  # Otherwise, it's already a string
    # Remove 'TwitterID' and 'dtype: object' lines
    tweet_lines = [line for line in tweet.split('\n') if not ('TwitterID' in line or 'dtype: object' in line)]
    tweet_text = '\n'.join(tweet_lines)
    print(f"Twitter id {index}:")
    print(f"{i}. {tweet_text}")
    print(f"\n>-Predicted Label: {clf.predict(X[df.index.get_loc(index), :])[0]}")
    print()

# Prompt the user to select a bully tweet for reporting
print("\nIndex numbers of bully tweets:")
for i, index in enumerate(bully_indices, start=1):
    print(f"{i}. Index: {index}")

while True:
    try:
        tweet_index = int(input("Which bully tweet you want to report (enter the index number): ")) - 1
        if tweet_index in range(len(bully_indices)):
            break
        else:
            print("Invalid input or index number not found in the bully tweets.")
    except ValueError:
        print("Invalid input. Please enter a valid index number.")

selected_tweet = df.loc[bully_indices[tweet_index], 'Tweet']
user_id = bully_indices[tweet_index]

# Construct the tweet message for reporting
politician_handle = "@annamalai_k"
report_message = f"@Twitter, this user {user_id} bully tweeted as'{selected_tweet}' and seems as bullying content about the politician {politician_handle}, please take action!"

# Post the tweet
try:
    response = client.create_tweet(text=report_message)
    print(f"Report tweet posted successfully!")
except Exception as e:
    print(f"Error posting report tweet: {e}")

