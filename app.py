from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import yfinance as yf
from transformers import pipeline
import spacy
import numpy as np
from scipy.optimize import minimize

app = Flask(__name__)

# Load NPL model. My laptop has GPU so I specify to run with CUDA
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)
sentiment_analyzer = pipeline("sentiment-analysis", device=0)

nlp = spacy.load("en_core_web_sm")

def ensure_complete_sentences(text):
    doc = nlp(text)
    final_sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text.endswith(('.', '!', '?')):
            final_sentences.append(sent_text)
        else:
            pass

    return " ".join(final_sentences)


def analyze_article(text):
    # Summarize the article
    try:
        raw_summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        summary = ensure_complete_sentences(raw_summary)
    except Exception as e:
        summary = "Error summarizing the article."
        print(f"Summarization error: {e}")

    # Extract entities
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        entities = []
        print(f"Entity extraction error: {e}")

    # Organizations
    org = []
    for entity, label in entities:
        if label == 'ORG' and entity not in org: 
            org.append(entity)
    if len(org) == 0:
        org = 'No organizations affected'

    # Analyze sentiment
    try:
        sentiment = sentiment_analyzer(text)[0]
    except Exception as e:
        sentiment = {"label": "Neutral", "score": 0}
        print(f"Sentiment analysis error: {e}")

    # Find stock
    try:
        df = pd.read_csv('company_tickers.csv')
    except Exception as e:
        print(f"Error loading ticker CSV: {e}")
        return {
            "summary": summary,
            "org": org,
            "sentiment": sentiment,
            "ticker": "Error loading ticker data",
            "current price": "N/A",
            "new price": "N/A"
        }

    ticker = []
    for entity, label in entities:
        for index, comp in df['company'].items():
            if entity.lower() == comp.lower():
                if df['ticker'][index] not in ticker:
                    ticker.append(df['ticker'][index]) 

    ### Stock price ###
    scaling_factor = 0.05
    percentage_change = scaling_factor * sentiment['score']

    if len(ticker) == 0:
        ticker = 'No stocks affected'
        current_price = ['N/A']
        new_price = ['N/A']  

    elif len(ticker) == 1:
        stock = yf.Ticker(ticker[0])  
        try:
            current_price = [stock.history(period="1d")["Close"].iloc[-1]]
            # Predicted price
            if sentiment['label'] == 'POSITIVE':
                new_price = [current_price[0] + (percentage_change * current_price[0])]
            elif sentiment['label'] == 'NEGATIVE':
                new_price = [current_price[0] - (percentage_change * current_price[0])]
            else:
                new_price = [current_price[0]]
        except Exception:
            current_price = ['N/A']
            new_price = ['N/A']

    elif len(ticker) > 1:
        current_price = []
        new_price = []
        for each_ticker in ticker:
            stock = yf.Ticker(each_ticker)
            try:
                each_current_price = stock.history(period="1d")["Close"].iloc[-1]
                current_price.append(each_current_price)
                if sentiment['label'] == 'POSITIVE':
                    new_price.append(each_current_price + (percentage_change * each_current_price))
                elif sentiment['label'] == 'NEGATIVE':
                    new_price.append(each_current_price - (percentage_change * each_current_price))
                else:
                    new_price.append(each_current_price)
            except Exception:
                current_price.append('N/A')
                new_price.append('N/A')

    # results
    return {
        "summary": summary,
        "org": org,
        "sentiment": sentiment,
        "ticker": ticker,
        "current price": current_price,
        "new price": new_price
    }

##############################################
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        article_text = request.form['article_text']
        results = analyze_article(article_text)
        return render_template('index.html', results=results, submitted=True)
    return render_template('index.html', submitted=False)

if __name__ == '__main__':
    app.run(debug=True)
