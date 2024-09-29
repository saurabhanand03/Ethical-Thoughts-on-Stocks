import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
from prophet import Prophet
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

st.title('Stock Analysis for Social and Environmental Good')

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

@st.cache_data
def load_esg_data():
    esg_data = pd.read_csv('data.csv')
    return esg_data


######################
### SELECT A STOCK ###
######################
START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
TOMORROW = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

# Load ESG data
esg_data = load_esg_data()

# Filter ESG data for desired grades
desired_grades = ['AAA', 'AA', 'A']
filtered_esg_data = esg_data[esg_data['total_grade'].isin(desired_grades)]

# Extract list of stock tickers with desired ESG grades
filtered_stock_list = filtered_esg_data['ticker'].unique()
filtered_stock_list = sorted([ticker.upper() for ticker in filtered_stock_list])

# Create a dropdown menu with the filtered stock list
selected_stock = st.selectbox('Select a stock:', filtered_stock_list)

# Load data for the selected stock
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
esg_data = load_esg_data()
data_load_state.empty()

##################
### STOCK DATA ###
##################
st.subheader('Raw data')

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    st.plotly_chart(fig)

plot_raw_data()

st.dataframe(data, hide_index=True)

##################
### ESG SCORES ###
##################
st.subheader(f'ESG Score for {esg_data[esg_data["ticker"] == selected_stock.lower()]["name"].iloc[0]}:')
st.info('What is an ESG Score? An ESG score is an objective measurement or evaluation of a given company, fund, or security\'s performance with respect to Environmental, Social, and Governance (ESG) issues.')

# Merge ESG ratings with stock data
esg_data = esg_data[esg_data['ticker'] == selected_stock.lower()]
esg_scores = esg_data[['environment_score', 'social_score', 'governance_score', 'total_score']].iloc[0]
reshaped_data = pd.DataFrame({
    'Category': ['environment', 'social', 'governance', 'total'],
    'Grade': [esg_data['environment_grade'].iloc[0], esg_data['social_grade'].iloc[0], esg_data['governance_grade'].iloc[0], esg_data['total_grade'].iloc[0]],
    'Level': [esg_data['environment_level'].iloc[0], esg_data['social_level'].iloc[0], esg_data['governance_level'].iloc[0], esg_data['total_level'].iloc[0]],
    'Score': [esg_data['environment_score'].iloc[0], esg_data['social_score'].iloc[0], esg_data['governance_score'].iloc[0], esg_data['total_score'].iloc[0]]
})

# Set 'Category' as the index
reshaped_data.set_index('Category', inplace=True)
# Display the reshaped DataFrame
st.dataframe(reshaped_data)

<<<<<<< HEAD
=======

##################
### STOCK DATA ###
##################
st.subheader('Raw data')

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    st.plotly_chart(fig)

plot_raw_data()

st.dataframe(data.tail(), hide_index=True)


>>>>>>> 9353962fd7f6ece7011159c53009eacfe15bb188
########################
### FINANCIAL RATIOS ###
########################
st.subheader('Financial Ratios')
# get ratios for the company
def getRatios(t):
    info = yf.Ticker(t).info
    #price/earnings ratio
    pe = info['forwardPE']
    if pe < 20:
        pe_eval = 'undervalued'
    elif pe >= 20 and pe < 25:
        pe_eval = 'fair'
    else:
        pe_eval = 'overvalued'
    
    #debt to equity ratio = total liabilities / shareholders equity
    bdf = yf.Ticker(t).balance_sheet
    se = bdf.loc['Stockholders Equity', bdf.columns[0]]
    l = bdf.loc['Total Debt',bdf.columns[0]]
    de = l/se
    if de <= 1.5:
        de_eval = 'undervalued'
    elif de > 1.5 and de < 2:
        de_eval = 'fair'
    else: #>=2
        de_eval = 'overvalued'
    
    return pd.DataFrame({'Price to Earnings':[pe,pe_eval],'Debt to Equity':[de,de_eval]},
                       index=['Ratio', 'Evaluation'])
# display the ratios for the selected stock and display its valuation
ratios = getRatios(selected_stock)
st.dataframe(ratios)

# walkthrough of the meaning for each of the ratios
st.write("Financial ratios are used to determine if a company's stocks are undervalued",
      "or overvalued based on company data. The price to earnings ratio",
      "compares stock price to the company's earnings. Meanwhile, the debt to equity ratio",
      "is the ratio of total liabilities to total stockholders' equity. The financial ratios", 
      "used for this project factor in fundamental accounting principles to draw conclusions",
      "about the financial bases for stock price valuation.")



# ########################
# ### PREDICTIVE MODEL ###
# ########################
# st.subheader('Model Performance')

# # Train RandomForestRegressor model
# def train(df, ticker):
#     ticker_df = df[df['Ticker'] == ticker]
#     X = ticker_df[['Open', 'High', 'Low', 'Volume']]
#     y = ticker_df['Adj Close']  # adjusted closing price
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
#     rf.fit(X_train, y_train)
#     predictions = rf.predict(X_test)
#     return y_test, predictions

# # Prepare data for training
# data['Ticker'] = selected_stock
# y_test, predictions = train(data, selected_stock)

# # Display model performance
# st.write(f'Mean Squared Error: {mean_squared_error(y_test, predictions)}')
# st.write(f'R^2 Score: {r2_score(y_test, predictions)}')

# # Plot actual vs predicted values
# def plot_predictions(y_test, predictions):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual'))
#     fig.add_trace(go.Scatter(x=y_test.index, y=predictions, mode='lines', name='Predicted'))
#     fig.layout.update(title_text='Actual vs Predicted Prices', xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)

# plot_predictions(y_test, predictions)


################################
### FORECASTING WITH PROPHET ###
################################
st.subheader('Forecasting')

# Predict forecast with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Show and plot forecast
st.write(forecast.tail())
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig1.layout.update(title_text='Forecast Plot', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

st.write('Forecast components:')
fig2 = m.plot_components(forecast)
st.write(fig2)


############################
### FORECASTING WITH RNN ###
############################
st.subheader('Forecasting with RNN')

model_load_state = st.text('Creating RNN model...')

def create_rnn_model(data_shape):
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, activation='tanh', input_shape=data_shape[1:]))
    model.add(SimpleRNN(128, return_sequences=False, activation='tanh'))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

# Sliding window approach with window of 30 days
X, Y = [], []
df = data['Close'].values
for i in range(len(data) - 30 - 1):
    X.append(df[i:(i + 30)])
    Y.append(df[i + 30])
X, Y = np.array(X).reshape(-1, 30, 1), np.array(Y)
# X[0] = days 1-30, Y[0] = day 31
# X[1] = days 2-31, Y[1] = day 32
# X[2] = days 3-32, Y[2] = day 33
# ...
# X[n] = days n+1 to n+30, Y[n] = day n+31
X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)

model = create_rnn_model(X_train.shape)
model.summary()

model_load_state.text('Fitting RNN model...')
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=100, batch_size=64)

model_load_state.text('Predicting next day price...')
X_pred = np.reshape(df[-30:], (1, 30, 1))
Y_pred = model.predict(X_pred)
predicted_price = Y_pred[0][0]

model_load_state.empty()

st.write(f'Predicted price for {TOMORROW}: ${predicted_price:.2f}')


########################
### FINANCIAL RATIOS ###
########################
# get ratios for the company
def getRatios(t):
    info = yf.Ticker(t).info
    #price/earnings ratio
    pe = info['forwardPE']
    if pe < 20:
        pe_eval = 'undervalued'
    elif pe >= 20 and pe < 25:
        pe_eval = 'fair'
    else:
        pe_eval = 'overvalued'
    
    #price/book = market cap / book value equity
    #book value equity = stockholders equity
    bdf = yf.Ticker('AAPL').balance_sheet
    se = bdf.loc['Stockholders Equity', bdf.columns[0]]
    pb = info['marketCap']/se
    if pb < 1:
        pb_eval = 'undervalued'
    elif pb == 1:
        pb_eval = 'fair'
    else:
        pb_eval = 'overvalued'
    
    #price/earnings to growth ratio
    peg = pe / info['earningsGrowth']
    if peg <= 1:
        peg_eval = 'undervalued'
    else:
        peg_eval = 'overvalued'
    
    #debt to equity ratio = total liabilities / shareholders equity
    l = bdf.loc['Total Debt',bdf.columns[0]]
    de = l/se
    if de <= 1.5:
        de_eval = 'undervalued'
    elif de > 1.5 and de < 2:
        de_eval = 'fair'
    else: #>=2
        de_eval = 'overvalued'
    
    return pd.DataFrame({'Price to Earnings':[pe,pe_eval],'Price to Books':[pb,pb_eval],
                         'Price/Earnings to Growth':[peg,peg_eval],'Debt to Equity':[de,de_eval]},
                       index=['Ratio', 'Evaluation'])
# display the ratios for the selected stock and display its valuation
ratios = getRatios(selected_stock)
st.dataframe(ratios)

# walkthrough of the meaning for each of the ratios
st.write("Financial ratios are used to determine if a company's stocks are undervalued",
      "or overvalued based on company data. The price to earnings ratio",
      "compares stock price to the company's earnings. The price to books ratio compares",
      "stock price to book value equity, or stockholders' equity. The price/earnings to",
      "growth ratio factors in the price earnings ratio and compares it to the earnings",
      "growth rate. Finally, the debt to equity ratio is the ratio of total liabilities"
      "to total stockholders' equity. The financial ratios used for this project factor",
      "in fundamental accounting principles to draw conclusions about the financial bases",
      "for stock price valuation.")


##################
### REFERENCES ###
##################
st.subheader('References')
st.markdown('''
- [Prophet](https://facebook.github.io/prophet/)
- [Keras](https://keras.io/)
- [Plotly](https://plotly.com/python/)
- [Streamlit](https://streamlit.io/)
- [Yahoo Finance](https://pypi.org/project/yfinance/)
''')
