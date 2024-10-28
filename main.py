import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta, date
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
import pickle
import time
import os
import finnhub
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# API keys
api_key_finnhub = 'cs0o249r01qru183mbbgcs0o249r01qru183mbc0'  # Insert your Finnhub API key here
fundamental_key = 'ISNFUN0Y5HKVSFST'  # Insert your Alpha Vantage API key here

# Title of the dashboard
st.title('Stock Dashboard')

# Initialize the model pipeline
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Function to get stock suggestions
@st.cache_data
def get_stock_suggestions():
    return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']

# Function to calculate EMA
def calculate_ema(data, period, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()

# Function to calculate MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# Function to calculate RSI
def calculate_rsi(data, periods=14, column='Close'):
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Initialize session state
if 'processed_articles' not in st.session_state:
    st.session_state.processed_articles = False

# Sidebar inputs with autocomplete
suggestions = get_stock_suggestions()
ticker = st.sidebar.selectbox('Ticker', suggestions, index=0)
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('today') - timedelta(days=28))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))

# EMA selection
ema_options = st.sidebar.multiselect(
    'Select EMA periods',
    ['EMA 9', 'EMA 18', 'EMA 27'],
    default=['EMA 9', 'EMA 18']
)


# Tabs for different sections
pricing_data, fundamental_data, news, news_research = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News", "News Research"])

# Pricing Data Tab
with pricing_data:
    # Download stock data from Yahoo Finance
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for ticker '{ticker}' in the selected date range.")
        else:
            # Plotting stock data with selected EMAs
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick",
                xaxis="x",
                yaxis="y"
            ))

            # Add line trace (initially invisible)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name="Line",
                line=dict(width=1),  # Make the line thinner
                visible=False
            ))

                # Add EMA traces
            for ema in ema_options:
                period = int(ema.split()[1])
                ema_values = calculate_ema(data, period)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=ema_values,
                    name=f'EMA {period}',
                    line=dict(width=1)  # Make the EMA lines thinner
                ))

            # Customize chart layout
            fig.update_layout(
                title=f"{ticker} Stock Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                dragmode='zoom',
                height=600,
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        x=0.5,
                        y=1.15,
                        buttons=list([
                            dict(
                                args=[{"visible": [True, False] + [True] * len(ema_options)}],
                                label="Candlestick",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": [False, True] + [True] * len(ema_options)}],
                                label="Line",
                                method="update"
                            )
                        ]),
                        bgcolor="blue",  # Set background color to dark
                        bordercolor="blue",  # Set a gray border
                        font=dict(
                            color="black"  # Set text color to white
                        )
                    )
                ]
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)


            # Simple recommendation based on short-term vs long-term moving averages
            short_ma = data['Close'].rolling(window=50).mean()
            long_ma = data['Close'].rolling(window=200).mean()

            if short_ma.iloc[-1] > long_ma.iloc[-1]:
                recommendation = "Buy"
            elif short_ma.iloc[-1] < long_ma.iloc[-1]:
                recommendation = "Sell"
            else:
                recommendation = "Hold"

            st.write(f"Simple Recommendation for {ticker}: {recommendation}")
            st.write("(Based on 50-day vs 200-day moving average)")

            # MACD Plot
            macd, signal, histogram = calculate_macd(data)
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=(f'{ticker} Stock Price', 'MACD'))
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'), row=1, col=1)
            fig_macd.add_trace(go.Scatter(x=data.index, y=macd, name='MACD'), row=2, col=1)
            fig_macd.add_trace(go.Scatter(x=data.index, y=signal, name='Signal Line'), row=2, col=1)
            fig_macd.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram'), row=2, col=1)
            fig_macd.update_layout(height=600, title_text="MACD Indicator")
            st.plotly_chart(fig_macd)

            # RSI Plot
            rsi = calculate_rsi(data)
            fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=(f'{ticker} Stock Price', 'RSI'))
            fig_rsi.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'), row=1, col=1)
            fig_rsi.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI'), row=2, col=1)
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig_rsi.update_layout(height=600, title_text="RSI Indicator")
            st.plotly_chart(fig_rsi)

    except Exception as e:
        st.error(f"Failed to retrieve data for ticker '{ticker}': {e}")

    if not data.empty:
        st.header('Price Movements')
        data2 = data.copy()
        data2['% Change'] = (data['Close'] / data['Close'].shift(1)) - 1
        data2.dropna(inplace=True)
        st.write(data2)

        # Calculate metrics
        annual_return = data2['% Change'].mean() * 252 * 100
        stdev = np.std(data2['% Change']) * np.sqrt(252)

        st.write(f'Annual Return: {annual_return:.2f}%')
        st.write(f'Standard Deviation: {stdev * 100:.2f}%')
        st.write(f'Risk Adjusted Return: {annual_return / (stdev * 100):.2f}')
    else:
        st.warning("No pricing data available for this ticker.")

# Fundamental Data Tab
with fundamental_data:
    st.header('Fundamental Data')
    fd = FundamentalData(fundamental_key, output_format='pandas')

    try:
        st.subheader('Balance Sheet')
        balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
        bs = balance_sheet.T[2:]
        bs.columns = list(balance_sheet.T.iloc[0])
        st.write(bs)

        st.subheader('Income Statement')
        income_statement = fd.get_income_statement_annual(ticker)[0]
        is1 = income_statement.T[2:]
        is1.columns = list(income_statement.T.iloc[0])
        st.write(is1)

        st.subheader('Cash Flow Statement')
        cash_flow = fd.get_cash_flow_annual(ticker)[0]
        cf = cash_flow.T[2:]
        cf.columns = list(cash_flow.T.iloc[0])
        st.write(cf)
    except Exception as e:
        st.error(f"Error fetching fundamental data: {e}")

# Stock News Tab
with news:
    st.header('Top 10 News')
    st.subheader(f'News for {ticker}')
    try:
        # Setup Finnhub client with your API key
        finnhub_client = finnhub.Client(api_key=api_key_finnhub)

        # Get news for the selected ticker
        try:
            news_list = finnhub_client.company_news(ticker, _from=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
            st.write(f"Total news articles found: {len(news_list)}")
        except Exception as e:
            st.error(f"Error fetching news data: {e}")
            news_list = []

        # Display the top 10 news
        for idx, news_item in enumerate(news_list[:10]):
            st.write(f"{idx + 1}. {news_item['headline']}")
            st.write(f"    {news_item['summary']}")
            st.write(f"    [Link to article]({news_item['url']})")
    except Exception as e:
        st.error(f"Error fetching news: {e}")


# News Research Tab
with news_research:
    st.header("News Research Tool 📈")
    
    # Dynamic URL input
    st.subheader("Article URLs")
    num_urls = st.number_input("Number of articles to analyze", min_value=1, max_value=10, value=3)
    
    urls = []
    for i in range(num_urls):
        url = st.text_input(f"URL {i+1}", key=f"url_{i}")
        if url:
            urls.append(url)
    
    process_url_clicked = st.button("Process URLs")
    file_path = "faiss_store_transformers.pkl"
    
    if process_url_clicked and urls:
        try:
            with st.spinner('Loading the FLAN-T5 model... Please wait...'):
                llm = load_model()
                embeddings = get_embeddings()
            st.success('Model loaded successfully!')
            
            with st.spinner('Processing articles...'):
                loader = UnstructuredURLLoader(urls=urls)
                article_data = loader.load()
                st.info(f"Loaded {len(article_data)} articles")
                
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(article_data)
                
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)
                
                st.session_state.processed_articles = True
                st.success("Articles processed successfully!")
        except Exception as e:
            st.error(f"Error processing articles: {str(e)}")
    elif process_url_clicked:
        st.warning("Please enter at least one URL")
    
    if st.session_state.processed_articles:
        st.subheader("Ask Questions About the Articles")
        query = st.text_input("Enter your question:")
        
        if query and os.path.exists(file_path):
            try:
                template = """Answer the following question based on the given context. If you don't know the answer, just say that you don't know.

                Context: {context}

                Question: {question}

                Answer:"""
                
                PROMPT = PromptTemplate(
                    template=template, 
                    input_variables=["context", "question"]
                )
                
                with open(file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                
                docs = vectorstore.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                llm = load_model()
                chain = LLMChain(llm=llm, prompt=PROMPT)
                
                with st.spinner('Generating answer...'):
                    result = chain.run(context=context, question=query)
                
                st.header("Answer")
                st.write(result)
                
                st.subheader("Sources:")
                for doc in docs:
                    st.write(doc.metadata.get('source', 'Unknown source'))
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
    else:
        st.info("Please process some articles first before asking questions.")