import streamlit as st

st.title("Stock Predictor")

stock = st.text_input("Enter stock symbol", "AAPL")

if st.button("Predict"):
    st.write("Fetching data and predicting...")
    # (Add your model logic here)
    st.line_chart([100, 102, 101, 105])  # Replace with real predictions
