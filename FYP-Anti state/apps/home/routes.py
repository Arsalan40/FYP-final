import csv
from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
from werkzeug.utils import secure_filename
from unicodedata import digit
import pickle
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
from flask import Flask, flash, request, redirect, render_template


@blueprint.route('/index')
@login_required
def index():
    return render_template('home/index.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


sentiment_analyzer_model = pickle.load(open('model.pkl', 'rb'))


def cleanText(x):
    x = x.encode('ascii', 'ignore').decode()  # remove emojis
    x = re.sub(r'https*\S+', '', x)  # remove urls
    x = re.sub(r'@\S+', '', x)  # remove mentions
    x = re.sub(r'#\S+', '', x)  # remove hashtags
    x = re.sub(r'\'w+', '', x)
    return x


def preporocess(tweet):
    tweet = [t for t in tweet if t not in string.digits]  # removing digits
    tweet = ''.join(tweet)
    tweet = [t for t in tweet if t not in string.punctuation]  # removing punctuations
    return ''.join(tweet)


@blueprint.route('/s_t', methods=["GET", "POST"])
def s_t():
    if request.method == "POST":
        f = request.files['file']
        limit = int(request.form['limit'])
        f.save(secure_filename(f.filename))
        flash('success file uploaded ')
        column_names = ['tweet', 'username', 'id']
        data = pd.read_csv(f.filename, names=column_names)

        data['tweet'] = data['tweet'].apply(
            preporocess)  # preprocessing the tweets removing stopwords , punctuations , digits
        data['tweet'] = data['tweet'].apply(lambda x: x.strip())  # removing spaces in the beginning of the tweet
        data['tweet'] = data['tweet'].dropna()



        preds = sentiment_analyzer_model.predict(data['tweet'][:limit])

        result = {}
        for i, u, d, j in zip(data['tweet'][:limit], data['username'][:limit], data['id'][:limit], preds):
            if j == 0.0:
                result[i] = (u, d, 'Non-AntiState')
            else:
                result[i] = (u, d, 'Anti State')

        df = pd.DataFrame(result)
        df.to_csv('filename.csv')
        return render_template("result.html", result=result)
