import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# %matplotlib inline
import re
import string
import nltk
from nltk.corpus import stopwords
# import gensim
# import pyLDAvis.gensim_models
# from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer

import time
# from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import pickle

warnings.filterwarnings('ignore')

columns = ['tweet', 'userid', 'username', 'created']
df = pd.read_csv('output.csv', names=columns, encoding='UTF_8')

#
# df.head()
#
#
# df.tail()
#
#
# df = df[['sentiment','text']]
#
# #4 in the sentiment column for positive tweets and 0 for negatives in the data set.
# #Let's change 4 to 1 and make the sentiment category of positive tweets "1".
#
# df['sentiment'] = df['sentiment'].replace(4,1)
# df.tail()
#
# df['sentiment'].value_counts()
#
#
# # df.info()


tweet, userid, username, created = list(df['tweet']), list(df['userid']), list(df['username']), list(df['created'])

print(tweet)


def clean_dataset(textdata):
    processedText = []
    wordLemm = WordNetLemmatizer()

    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-z0-9#]"
    stopwords_set = stopwords.words('english')

    for tweet in textdata:
        tweet = tweet.lower()
        tweet = re.sub(urlPattern, ' URL', tweet)
        tweet = re.sub(userPattern, ' USER', tweet)
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = " ".join([i for i in tweet.split() if i not in stopwords_set])

        tweetwords = ''
        for word in tweet.split():
            if len(word) > 1:
                word = wordLemm.lemmatize(word)
                tweetwords += (word + ' ')
        processedText.append(tweetwords)

    return processedText


t = time.time()
processedtext = clean_dataset(tweet)
print(f'Time Taken: {round(time.time() - t)} seconds')

# processedtext

# processedtext[1:5]

# data_neg = " ".join(processedtext[:800000])
# plt.figure(figsize = (15,15))
# wc = WordCloud(max_words=400, width=1000, height=500, max_font_size=100, collocations=False).generate(data_neg)
# plt.imshow(wc)
# plt.axis('off')
# plt.show()


# data_pos = " ".join(processedtext[800000:])
# wc = WordCloud(max_words=400, width=1000, height=500, max_font_size=100, collocations=False).generate(data_pos)
# plt.figure(figsize = (15,15))
# plt.imshow(wc)
# plt.axis('off')
# plt.show()


df['clean_tweet'] = processedtext


def hashtag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags


# df
#
#
# ht_positive = hashtag_extract(df['clean_tweet'][df['sentiment']==1])
#
# ht_negative = hashtag_extract(df['clean_tweet'][df['sentiment']==0])
#
#
# ht_positive = sum(ht_positive, [])
# ht_negative = sum(ht_negative, [])
#
#
# freq = nltk.FreqDist(ht_positive)
# d = pd.DataFrame({'Hashtag': list(freq.keys()),'Count': list(freq.values())})
# d.head()
#
#
# #
# # d = d.nlargest(columns='Count', n=15)
# # plt.figure(figsize=(15,9))
# # sns.barplot(data=d, x='Hashtag', y='Count')
# # plt.show()
#
#
#
# freq = nltk.FreqDist(ht_negative)
# d = pd.DataFrame({'Hashtag': list(freq.keys()),
#                  'Count': list(freq.values())})
# d.head()
#
# #
# # d = d.nlargest(columns='Count', n=15)
# # plt.figure(figsize=(15,9))
# # sns.barplot(data=d, x='Hashtag', y='Count')
# # plt.show()
#
#
# #from sklearn.feature_extraction.text import CountVectorizer
# #bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# #bow = bow_vectorizer.fit_transform(df['clean_tweet'])
#
# #from sklearn.model_selection import train_test_split
# #x_train, x_test, y_train, y_test = train_test_split(bow, df['sentiment'], random_state=42, test_size=0.25)
#
#
# #from sklearn.linear_model import LogisticRegression
# #from sklearn.metrics import f1_score, accuracy_score
# #model = LogisticRegression()
# #model.fit(x_train, y_train)
# #pred = model.predict(x_test)
# #f1_score(y_test, pred)
#
#
# #accuracy_score(y_test,pred)
#
# X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment, test_size = 0.20, random_state = 1)
#
#
# vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
# vectoriser.fit(X_train)
#
#
# TfidfVectorizer(max_features=500000, ngram_range=(1, 2))
#
# X_train = vectoriser.transform(X_train)
# X_test  = vectoriser.transform(X_test)
#
#
# def model_Evaluate(model):
#
#     y_pred = model.predict(X_test)
#
#     print(classification_report(y_test, y_pred))
#
#     cf_matrix = confusion_matrix(y_test, y_pred)
#
#     categories  = ['Negative','Positive']
#     group_names = ['True Neg','False Pos', 'False Neg','True Pos']
#     group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
#
#     labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
#     labels = np.asarray(labels).reshape(2,2)
#
#     sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
#                 xticklabels = categories, yticklabels = categories)
#
#     # plt.xlabel("Predicted values", fontdict = {'size':10}, labelpad = 10)
#     # plt.ylabel("Actual values"   , fontdict = {'size':10}, labelpad = 10)
#     # plt.title ("Confusion Matrix", fontdict = {'size':15}, pad = 20)
#
#
# BNBmodel = BernoulliNB(alpha = 2)
# BNBmodel.fit(X_train, y_train)
# acc_BNB= model_Evaluate(BNBmodel)
#
#
#
# file = open('vectoriser-ngram-(1,2).pickle','wb')
# pickle.dump(vectoriser, file)
# file.close()
# file = open('Sentiment-BNB.pickle','wb')
# pickle.dump(BNBmodel, file)
# file.close()
#
#
# def load_models():
#
#     file = open('vectoriser-ngram-(1,2).pickle', 'rb')
#     vectoriser = pickle.load(file)
#     file.close()
#
#     file = open('Sentiment-BNBv1.pickle', 'rb')
#     BNBmodel = pickle.load(file)
#     file.close()
#
#     return vectoriser, BNBmodel
#
# def predict(vectoriser, model, text):
#
#     textdata = vectoriser.transform(clean_dataset(text))
#     sentiment = model.predict(textdata)
#
#
#     data = []
#     for text, pred in zip(text, sentiment):
#         data.append((text,pred))
#
#
#     df = pd.DataFrame(data, columns = ['text','sentiment'])
#     df = df.replace([0,1], ["Negative","Positive"])
#     return df
#
if __name__ == "__main__":
    text = pd.read_csv("output.csv")
    test_text = text.to_string()
    # df = predict(vectoriser, BNBmodel, test_text)
    df.to_csv("Result.csv")
    print(df.head())
