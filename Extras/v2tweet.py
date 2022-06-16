import tweepy
# your bearer token
MY_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAADKvaAEAAAAAu1XlCzwc3FHXuSwzmI1k6M18fZg%3DBaIZIwyyLNDPIqfnzMXsfIepuEWeESSCBzeeGwNySMx8mR4lxI"
# create your client
client = tweepy.Client(bearer_token=MY_BEARER_TOKEN)

# search_query = "#imran khan -in:retweets"

# query to search for tweets
query = "#imran khan lang:en -is:retweet"
# your start and end time for fetching tweets

# get tweets from the API
tweets = client.search_recent_tweets(query=query,

                                     tweet_fields = ["created_at", "text", "source"],
                                     user_fields = ["name", "username", "location", "verified", "description"],
                                     max_results = 100,
                                     expansions='author_id'
                                     )

# tweet specific info
print(len(tweets.data))
# user specific info
print(len(tweets.includes["users"]))

# import the pandas library
import pandas as pd
# create a list of records
tweet_info_ls = []
# iterate over each tweet and corresponding user details
for tweet, user in zip(tweets.data, tweets.includes['users']):
    tweet_info = {

        'text': tweet.text,

        'name': user.name,
        'username': user.username,

    }
    tweet_info_ls.append(tweet_info)
# create dataframe from the extracted records
tweets_df = pd.DataFrame(tweet_info_ls)

tweets_df.to_csv("Data.csv")
# display the dataframe
print(tweets_df.head())








