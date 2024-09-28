import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go

START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Analysis and Prediction App')

holdings = ['AMZN', 'COST', 'WMT', 'HD', 'LOW', 'TJX', 'ORLY', 'MCK', 'CVS', 'TGT']
selected_stock = st.selectbox('Select stock for analysis', holdings)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

@st.cache_data
def load_esg_data():
    esg_data = pd.read_csv('data.csv')
    return esg_data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
esg_data = load_esg_data()
data_load_state.text('Loading data... done!')

# Merge ESG ratings with stock data
esg_data = esg_data[esg_data['ticker'].str.upper() == selected_stock]
esg_scores = esg_data[['environment_score', 'social_score', 'governance_score', 'total_score']].iloc[0]

# Add ESG scores to the data for model training
data['environment_score'] = esg_scores['environment_score']
data['social_score'] = esg_scores['social_score']
data['governance_score'] = esg_scores['governance_score']
data['total_score'] = esg_scores['total_score']

st.subheader('Raw data')
st.write(data.drop(columns=['environment_score', 'social_score', 'governance_score', 'total_score']))

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Train RandomForestRegressor model
def train(df, ticker):
    ticker_df = df[df['Ticker'] == ticker]
    X = ticker_df[['Open', 'High', 'Low', 'Volume', 'environment_score', 'social_score', 'governance_score', 'total_score']]
    y = ticker_df['Adj Close']  # adjusted closing price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    return y_test, predictions

# Prepare data for training
data['Ticker'] = selected_stock
y_test, predictions = train(data, selected_stock)

# Display model performance
st.subheader('Model Performance')
st.write(f'Mean Squared Error: {mean_squared_error(y_test, predictions)}')
st.write(f'R^2 Score: {r2_score(y_test, predictions)}')

# Plot actual vs predicted values
def plot_predictions(y_test, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=y_test.index, y=predictions, mode='lines', name='Predicted'))
    fig.layout.update(title_text='Actual vs Predicted Prices', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_predictions(y_test, predictions)