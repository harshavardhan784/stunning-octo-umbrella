# Stock Dashboard 🚀

A comprehensive stock analysis dashboard built with Streamlit that combines technical analysis, fundamental data, news tracking, and AI-powered research capabilities.

## 🌟 Features

### 📊 Pricing Data Analysis
* Interactive candlestick/line chart with customizable EMAs
* Technical indicators (MACD, RSI)
* Price movement analytics
* Trading recommendations

### 📈 Fundamental Data
* Balance Sheet
* Income Statement
* Cash Flow Statement

### 📰 News Tracking
* Real-time top 10 news
* Article summaries and links
* Date-range filtering

### 🤖 AI-Powered News Research
* Multi-article processing
* Natural language querying
* FLAN-T5 powered responses
* Source attribution

## 🛠️ Installation

```bash
# Clone repository
git clone <repository-url>
cd stock-dashboard

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

```txt
streamlit
pandas
numpy
yfinance
plotly
alpha_vantage
stocknews
finnhub-python
transformers
langchain
faiss-cpu
sentence-transformers
unstructured
```

## 🔑 API Setup

1. Get API keys from:
   * Finnhub
   * Alpha Vantage

2. Replace in code:
```python
api_key_finnhub = 'YOUR_FINNHUB_API_KEY'
fundamental_key = 'YOUR_ALPHA_VANTAGE_API_KEY'
```

## 🚀 Usage

```bash
streamlit run app.py
```

### Dashboard Navigation

1. **Sidebar Controls**
   * Stock ticker selection
   * Date range picker
   * EMA period selection

2. **Main Tabs**
   * Pricing Data
   * Fundamental Data
   * Top 10 News
   * News Research

## 📊 Technical Indicators

### MACD
* Fast period: 12
* Slow period: 26
* Signal period: 9

### RSI
* Period: 14
* Overbought: 70
* Oversold: 30

### EMA
* Customizable periods
* Default: 9, 18, 27 days

## 🔍 News Research Guide

1. Input article URLs (max 10)
2. Click "Process URLs"
3. Ask questions
4. Get AI-powered answers

## 🚀 Performance Features

* Model caching
* Efficient data processing
* Memory optimization
* Vectorized calculations

## 🙏 Acknowledgments

* Streamlit framework
* Yahoo Finance
* Finnhub
* Alpha Vantage
* FLAN-T5
* LangChain
