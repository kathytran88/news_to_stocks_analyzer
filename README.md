# News to stock analyzer
## Overview:
Create a webpage that receives users' input (text of news articles) and outputs:
- A summary of the article<br>
- Sentiment report<br>
- Companies affected by the news<br>
- Their current stock prices<br>
- Predicted stock prices of companies after the news

## Description
- Integrated NLP models using Hugging Face Transformers to summarize news articles, perform sentiment analysis, and identify affected organizations using SpaCy's NER (Named Entity Recognition).<br>
- Predicted the stock market impact of news articles by correlating sentiment scores with stock price adjustments<br>
- Leveraged yfinance API to fetch and analyze stock data.
