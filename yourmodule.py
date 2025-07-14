# yourmodule.py

import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np

# -------------------------
# PHASE 2: YouTube Comments
# -------------------------

def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None

def get_comments(video_id, api_key, max_comments=100):
    from googleapiclient.discovery import build

    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments[:max_comments]

# -------------------------
# PHASE 3: Sentiment (VADER)
# -------------------------

analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# -------------------------
# PHASE 4: Toxicity (TensorFlow)
# -------------------------

tox_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
tox_model = TFAutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert", from_pt=True)

def classify_toxicity_tf(text):
    inputs = tox_tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    logits = tox_model(inputs.data)[0]
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    return "Toxic" if probs[1] > 0.5 else "Non-Toxic"

# -------------------------
# PHASE 5: Emotion (GoEmotions)
# -------------------------

emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
    'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

emo_tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
emo_model = TFAutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")

def detect_emotion(text):
    inputs = emo_tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    logits = emo_model(inputs.data)[0]
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    max_index = np.argmax(probs)
    return emotion_labels[max_index]