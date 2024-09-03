import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def prepare_data(stock_data):
    df = stock_data.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    return df

def train_model(df):
    X = df[['Days']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def make_predictions(model, df, days_to_predict=30):
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Days'] = (future_df['Date'] - df['Date'].min()).dt.days
    
    future_predictions = model.predict(future_df[['Days']])
    future_df['Predicted_Close'] = future_predictions
    
    return future_df

def create_plot(df, future_df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'], mode='lines', name='Predicted Data', line=dict(dash='dash')))
    
    fig.update_layout(title='Stock Price Trend and Prediction',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      template='plotly_white')
    
    return fig.to_json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    df = prepare_data(stock_data)
    model, X_test, y_test = train_model(df)
    future_df = make_predictions(model, df)
    
    plot_json = create_plot(df, future_df)
    
    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mse)
    
    return jsonify({'plot': plot_json, 'rmse': rmse})

if __name__ == '__main__':
    app.run(debug=True)