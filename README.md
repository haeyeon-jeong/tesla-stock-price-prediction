# Tesla Stock Price Prediction for January 2024

## Overview
This project explores Tesla (TSLA) stock price forecasting using historical market data and time-series feature engineering.

Models were developed using lagged stock prices, moving averages (SMA/EMA), and trading indicators, with additional NLP features derived from Tesla-related news articles scraped from NBC News and The New York Times. 

The study compares stock-only and news-enhanced models to evaluate whether textual information improves short-term price prediction.

---

## Authors
Haeyeon Jeong 

---

## Data Source
Stock Market Data: 
- Yahoo Finance (yfinance)
- Tesla (TSLA)
- Period: January 2023 – January 2024
  
News Data (Custom-built news collection pipeline):
- NBC News Tesla articles (Selenium scraper)
- New York Times Article Search API
- Tesla, Cybertruck, and Elon Musk related news
The scraped datasets were cleaned, deduplicated, merged, and filtered into a unified Tesla news corpus.
  
---

## Methodology

The pipeline includes:

### 1. Data Collection  
   - Download Tesla historical stock prices using Yahoo Finance
   - Scrape Tesla-related news articles from NBC News
   - Collect business and technology articles from The New York Times API
   - Merge and clean all news sources

### 2. NLP Processing
   - Create combined article text using titles and summaries
   - Generate TF-IDF representations
   - Calculate sentiment scores using:
     - VADER
     - TextBlob
   - Aggregate news features by day

### 3. Time-Series Feature Engineering  
Generate:
   - Lagged closing prices
   - SMA (10-day, 20-day)
   - EMA (10-day, 20-day)
   - Price range metrics
   - Volume-based indicators

### 4. Model Training
Train and compare:
   - Baseline stock-only regression models
   - Sentiment-enhanced models
   - TF-IDF-enhanced models

### 5. Evaluation
Evaluate on:
   - 2023 test split
   - January 2024 forecasting period
         
Metrics:
   - R²
   - MSE
     
---

## Results
| Model          | January 2024 R² |
| -------------- | --------------- |
| Base Model     | -0.564          |
| SMA-Based Lag  | 0.9346          |
| EMA-Based Lag  | 0.9452           |
| TF-IDF + EMA   | **0.9503**       |
| TextBlob + EMA | 0.9451           |
| VADER + EMA    | 0.9463           |

The TF-IDF + EMA model achieved the best January 2024 forecasting performance, demonstrating that textual news information can provide incremental predictive value beyond historical stock prices alone.

---

## Project Structure
```
tesla-stock-price-prediction/
│
├── codes/
│   ├── tesla_stock_prediction.ipynb (notebook version)
│   └── tesla_stock_prediction.py
│
├── data/
│   ├── combined_tesla_news_clean.csv
│   ├── nbc_tesla_news_clean.csv
│   └── nyt_tesla_multiquery_news.csv
│
├── scraping/
│   ├── nbc_tesla_news_scrape.py
│   ├── nytimes_tesla_news_scrape.py
│   └── tesla_news_filtered.py
│
├── report/
│   └── Tesla Stock Price Prediction.pdf
│
└── README.md
```

---

## How to Run

### Install Required Packages

```bash
pip install yfinance pandas numpy scikit-learn matplotlib nltk textblob vaderSentiment
```

### Run the Project

The repository includes the cleaned news dataset (`combined_tesla_news_clean.csv`).

Run the main forecasting pipeline:

```bash
python tesla_stock_prediction.py
```

The script will:

* Download Tesla stock data from Yahoo Finance
* Load the cleaned Tesla news dataset
* Generate lag, SMA, EMA, sentiment, and TF-IDF features
* Train stock-only and news-enhanced forecasting models
* Evaluate model performance and generate comparison plots

---

## Key Insights

- EMA-based lag features achieved the strongest forecasting performance.
- Historical price trends were the most important predictors of Tesla stock prices.
- TF-IDF news features provided modest improvements in short-term forecasting accuracy.
- General news coverage contributed less predictive value than market-based indicators in this dataset.

---

## Future Improvements

- Fine-tune FinBERT for financial sentiment analysis.
- Experiment with XGBoost and LightGBM.
- Add market event features and earnings announcements.
- Deploy real-time prediction pipeline.
