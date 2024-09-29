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
from keras.metrics import MeanSquaredError
import io
import contextlib

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
TOMORROW = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

# Load ESG data
esg_data = load_esg_data()

# # Filter ESG data for desired grades
# desired_grades = ['AAA', 'AA', 'A']
# filtered_esg_data = esg_data[esg_data['total_grade'].isin(desired_grades)]

# # Extract list of stock tickers with desired ESG grades
# filtered_stock_list = filtered_esg_data['ticker'].unique()
# filtered_stock_list = sorted([ticker.upper() for ticker in filtered_stock_list])

# All stocks
stock_list = esg_data['ticker'].unique()
stock_list = sorted([ticker.upper() for ticker in stock_list])

# Create a dropdown menu with the filtered stock list
selected_stock = st.selectbox('Select a stock:', stock_list)

# Load data for the selected stock
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
esg_data = load_esg_data()
data_load_state.empty()


##################
### STOCK DATA ###
##################
st.subheader(f'Stock data for {esg_data[esg_data["ticker"] == selected_stock.lower()]["name"].iloc[0]}')

# Plot stock data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
fig.update_layout(margin=dict(t=0, b=0))
st.plotly_chart(fig)

# st.dataframe(data, hide_index=True)
# st.dataframe(data.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x), hide_index=True)
formatted_data = data.style.format({col: "{:.2f}" for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']})
st.dataframe(formatted_data, hide_index=True, use_container_width=True)


##################
### ESG SCORES ###
##################
st.subheader('ESG Score:')
st.info('Invest in Your Values: Our ESG Stock Indicator empowers clients to make'+ 
        'investment decisions that reflect their values and beliefs. By focusing on'+
        'companies with strong Environmental, Social, and Governance practices, clients'+
        'can grow their portfolio while supporting a sustainable and equitable future for'+
        'all communities.')

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


########################
### FINANCIAL RATIOS ###
########################
st.subheader('Financial Ratios')

# walkthrough of the meaning for each of the ratios
st.info("Financial ratios are used to determine if a company's stocks are undervalued "+
      "or overvalued based on financial data. These ratios use fundamental accounting principles "+
      "to draw conclusions about stock price valuation. The price to earnings ratio "+
      "compares stock price to the company's earnings. The price to books ratio compares "+
      "stock price to book value equity, which is the same as stockholders' equity. "+
      "The price/earnings to growth ratio factors in the price earnings ratio and "+
      "compares it to the earnings growth rate. Finally, the debt to equity ratio "+
      "is the ratio of total liabilities to total stockholders' equity. ")

# get ratios for the company
def getRatios(t):
    i = yf.Ticker(t).info
    #price/earnings ratio
    pe = i['forwardPE']
    if pe < 20:
        pe_eval = 'undervalued'
    elif pe >= 20 and pe < 25:
        pe_eval = 'fair'
    else:
        pe_eval = 'overvalued'
    
    #price/book = market cap / book value equity
    #book value equity = stockholders equity
    bdf = yf.Ticker(t).balance_sheet
    se = bdf.loc['Stockholders Equity', bdf.columns[0]]
    pb = i['marketCap']/(se*100)
    if pb < 1:
        pb_eval = 'undervalued'
    elif pb == 1:
        pb_eval = 'fair'
    else:
        pb_eval = 'overvalued'
        print("ASDfad")
    
    #price/earnings to growth ratio
    peg = pe / (i['earningsGrowth']*100)
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


############################
### FORECASTING WITH RNN ###
############################
st.subheader('Forecasting with RNN')

model_create_state = st.text('Creating RNN model...')

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def create_rnn_model(data_shape):
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, activation='tanh', input_shape=data_shape[1:]))
    model.add(SimpleRNN(128, return_sequences=False, activation='tanh'))
    model.add(Dense(32))
    model.add(Dense(1))
    return model

model = create_rnn_model(X_train.shape)
model_summary = io.StringIO()
with contextlib.redirect_stdout(model_summary):
    model.summary()

st.code(model_summary.getvalue())
model_create_state.empty()

model_fit_state = st.text('Fitting RNN model...')
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanSquaredError()])
# model.fit(X_train, Y_train, epochs=100, batch_size=64)
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test))
model_fit_state.empty()

# Evaluate the model
model_evaluate_state = st.text('Evaluating RNN model...')
train_loss, train_mse = model.evaluate(X_train, Y_train, verbose=0)
test_loss, test_mse = model.evaluate(X_test, Y_test, verbose=0)
model_evaluate_state.empty()

st.write(f"Train MSE: {train_mse:.4f}")
st.write(f"Test MSE: {test_mse:.4f}")

model_predict_state = st.text('Predicting next day price...')
X_pred = np.reshape(df[-30:], (1, 30, 1))
Y_pred = model.predict(X_pred)
predicted_price = Y_pred[0][0]
model_predict_state.empty()

predicted_price_df = pd.DataFrame({
    'Date': [TOMORROW],
    'Close': [f"{predicted_price:.2f}"]
})
st.dataframe(predicted_price_df, hide_index=True)


# ################################
# ### FORECASTING WITH PROPHET ###
# ################################
# st.subheader('Forecasting with Prophet')

# # Predict forecast with Prophet
# df_train = data[['Date', 'Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=365)
# forecast = m.predict(future)

# # Plot forecast stock data
# fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
# fig1.update_layout(margin=dict(t=0, b=0))
# st.plotly_chart(fig1)

# formatted_forecast = data.style.format({col: "{:.2f}" for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']})
# st.dataframe(forecast, hide_index=True, use_container_width=True)

# # st.write('Forecast components:')
# # fig2 = m.plot_components(forecast)
# # st.write(fig2)


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

st.subheader('Credits')
st.write('Saurabh Anand, Anjali Krishna, Neha Jupalli, Andria Gonzalez Lopez')