import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
import investpy2

def get_moneycontrol_data(ticker):
    try:
        url = f"https://www.moneycontrol.com/india/stockpricequote/{ticker.lower()}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        price_tag = soup.find('div', class_='inprice1 nsecp')
        price = price_tag.text.strip() if price_tag else "N/A"
        return price
    except:
        return "Error fetching data"

def get_investing_data(ticker):
    try:
        search_results = investpy2.search_quotes(text=ticker, products=['stocks'], countries=['india'])
        historical = search_results.retrieve_historical_data(from_date='01/01/2022', to_date='01/01/2024')
        return historical
    except Exception as e:
        st.error(str(e))
        return None

st.set_page_config(layout="wide")
st.title("📈 Indian Stock Market Predictor with TradingView")

ticker = st.text_input("Enter NSE stock symbol (e.g., RELIANCE)", "RELIANCE")
yf_ticker = f"{ticker.upper()}.NS"

st.markdown(f"""
<iframe src="https://s.tradingview.com/embed-widget/mini-symbol-overview/?locale=en#%7B%22symbol%22%3A%22NSE%3A{ticker.upper()}%22%2C%22width%22%3A%22100%25%22%2C%22height%22%3A%22300%22%2C%22isTransparent%22%3Atrue%2C%22colorTheme%22%3A%22dark%22%2C%22autosize%22%3Atrue%7D"
width="100%" height="300" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
""", unsafe_allow_html=True)

source = st.selectbox("Select Data Source", ["Yahoo Finance", "Moneycontrol", "Investing.com"])

if source == "Yahoo Finance":
    data = yf.download(yf_ticker, start="2019-01-01", end="2024-01-01")
elif source == "Moneycontrol":
    price = get_moneycontrol_data(ticker)
    st.success(f"Current Price from Moneycontrol: {price}")
    data = None
elif source == "Investing.com":
    data = get_investing_data(ticker)
else:
    data = None

if data is not None and not data.empty:
    data["Target"] = data["Close"].shift(-1)
    features = []
    labels = []
    for i in range(len(data)-5):
        features.append(data["Close"].values[i:i+5])
        labels.append(data["Target"].values[i+5-1])

    X = np.array(features)
    y = np.array(labels)

    model = LinearRegression()
    model.fit(X[:-20], y[:-20])
    predictions = model.predict(X[-20:])

    df_result = pd.DataFrame({
        "Predicted": predictions,
        "Actual": y[-20:]
    }).reset_index(drop=True)

    st.subheader("📊 Prediction vs Actual (Last 20 Days)")
    st.line_chart(df_result)
