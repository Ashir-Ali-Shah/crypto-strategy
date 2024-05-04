import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(page_title="Top Cryptocurrencies Analysis", layout="wide", page_icon="📈")

# Title of the Streamlit app
st.title("Top Cryptocurrencies by Trading Volume")
st.markdown("Explore the top 15 cryptocurrencies based on their trading volumes and year-over-year performance.")

# Sidebar for input to keep main area less cluttered
with st.sidebar:
    year = st.selectbox("Select Year", options=range(2023, 2019, -1), index=0)

# Define the start and end dates for the given year
start_date = dt.datetime(year, 1, 1)
end_date = dt.datetime(year, 12, 31)

# Define an extended list of cryptocurrencies (to ensure a broad selection for ranking)
cryptos = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'LTC-USD', 'EOS-USD', 'BNB-USD',
    'XTZ-USD', 'XLM-USD', 'LINK-USD', 'TRX-USD', 'NEO-USD', 'IOTA-USD', 'DASH-USD',
    'DOT-USD', 'UNI-USD', 'DOGE-USD', 'SOL-USD', 'AVAX-USD', 'FIL-USD', 'AAVE-USD',
    'ALGO-USD', 'ATOM-USD', 'VET-USD', 'ICP-USD', 'FTT-USD', 'SAND-USD', 'AXS-USD',
    'MATIC-USD', 'THETA-USD', 'XTZ-USD', 'EGLD-USD', 'KSM-USD', 'CAKE-USD', 'MKR-USD',
    'COMP-USD', 'ZEC-USD', 'XMR-USD', 'KCS-USD', 'HT-USD', 'OKB-USD', 'LEO-USD',
    'WAVES-USD', 'MIOTA-USD', 'LUNA1-USD', 'NEAR-USD', 'APE-USD', 'GMT-USD', 'GRT-USD'
]

# Fetch historical data
@st.cache_data(show_spinner=False)  # Cache the data for performance using the updated caching mechanism
def fetch_crypto_data(cryptos, start, end):
    crypto_data = {}
    for crypto in cryptos:
        data = yf.download(crypto, start=start, end=end)
        crypto_data[crypto] = data[['Adj Close', 'Volume']]
    return crypto_data

crypto_data = fetch_crypto_data(cryptos, start_date, end_date)

# Calculate average trading volume and price change for each cryptocurrency
average_volumes = {}
price_changes = {}
for crypto, data in crypto_data.items():
    if not data.empty:
        average_volumes[crypto] = data['Volume'].mean()
        price_changes[crypto] = ((data['Adj Close'][-1] - data['Adj Close'][0]) / data['Adj Close'][0]) * 100

# Sort cryptos by average volume in descending order and select top 15
top_cryptos = sorted(average_volumes, key=average_volumes.get, reverse=True)[:15]

# Display top cryptocurrencies, their average volume, and price change in a nicer format using metrics
st.subheader("Top 15 Cryptocurrencies by Trading Volume")
col1, col2, col3 = st.columns(3)
for i, crypto in enumerate(top_cryptos):
    with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
        st.metric(label=crypto,
                  value=f"{average_volumes[crypto]:,.0f}",
                  delta=f"{price_changes.get(crypto, 0):.2f}% change")

# Visualization of the price data of the top cryptocurrencies with improved aesthetics
expander = st.expander("View Detailed Price Charts")
with expander:
    col1, col2 = st.columns(2)
    for i, crypto in enumerate(top_cryptos):
        with (col1 if i % 2 == 0 else col2):
            fig, ax = plt.subplots()
            ax.plot(crypto_data[crypto]['Adj Close'], label=f'{crypto} Adjusted Close', color='purple')
            ax.set_title(f"{crypto} Adjusted Close Price in {year}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Adjusted Close Price (USD)")
            ax.legend()
            st.pyplot(fig)

# Fetch data from yfinance
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# Simulate fetching data for the last 5 years
start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
price_data = fetch_data(cryptos, start_date, end_date)

# Normalize the date index
price_data.index = pd.to_datetime(price_data.index)

# Placeholder for market cap
market_caps = price_data.copy()

# Define portfolio strategy functions
def market_cap_weighted_allocations(df, date):
    day_data = df.loc[date]
    total_market_cap = day_data.sum()
    weights = day_data / total_market_cap
    return weights

def capped_market_cap_weighted_allocations(df, date, cap=0.25):
    weights = market_cap_weighted_allocations(df, date)
    weights = weights.clip(upper=cap)
    weights /= weights.sum()
    return weights

# Placeholder Streamlit app
st.title('Crypto ETF Investment Simulator')

# Strategy selection
strategy_options = {
    'Market Cap Weighted': market_cap_weighted_allocations,
    'Capped Market Cap Weighted': capped_market_cap_weighted_allocations,
}
selected_strategy = st.sidebar.selectbox('Select Portfolio Strategy', list(strategy_options.keys()))

# Date selection for historical data viewing, limit max date to January 1st, 2024
max_date = datetime(year, 1, 1)
date = datetime(year, 1, 1)

# Data processing based on selection
def process_data(strategy_func):
    date_str = date.strftime('%Y-%m-%d')
    if pd.to_datetime(date_str) in price_data.index:
        weights = strategy_func(market_caps, pd.to_datetime(date_str))
        return weights
    else:
        st.write("No data available for the selected date.")
        return pd.DataFrame()

weights = process_data(strategy_options[selected_strategy])

# Display weights in a table
if not weights.empty:
    st.write(f"Weights for {selected_strategy} on {date}:")
    st.dataframe(weights)
    # Get market cap values for the selected date if available
    if pd.to_datetime(date) in market_caps.index:
        market_cap_values = market_caps.loc[pd.to_datetime(date)].rename("Market Cap")
        st.line_chart(market_cap_values)  # Plot market cap values
    else:
        st.write("No market cap data available for the selected date.")
else:
    st.write("No weights to display.")


# Function to calculate cumulative returns
def cumulative_returns(weights, price_data):
    returns = price_data.pct_change().dropna()
    weighted_returns = (weights * returns).sum(axis=1)
    cumulative = (1 + weighted_returns).cumprod() - 1
    return cumulative

# Function to calculate Sharpe ratio (Assuming risk-free rate = 0 for simplification)
def sharpe_ratio(weights, price_data):
    returns = price_data.pct_change().dropna()
    weighted_returns = (weights * returns).sum(axis=1)
    return np.mean(weighted_returns) / np.std(weighted_returns)

# Function to calculate annualized volatility
def annualized_volatility(weights, price_data):
    returns = price_data.pct_change().dropna()
    weighted_returns = (weights * returns).sum(axis=1)
    return np.std(weighted_returns) * np.sqrt(252)  # 252 trading days

# Assuming you have fetched and processed your price data and weights
# Assuming you have fetched and processed your price data and weights
if not weights.empty:
    cum_returns = cumulative_returns(weights, price_data)
    if not cum_returns.empty:
        sharpe = sharpe_ratio(weights, price_data)
        volatility = annualized_volatility(weights, price_data)

        st.metric("Cumulative Returns", f"{cum_returns.iloc[-1]:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Annualized Volatility", f"{volatility:.2%}")

        st.line_chart(cum_returns)  # Plot cumulative returns
    else:
        st.write("No cumulative returns data to display.")
else:
    st.write("No weights data to display.")


def get_data(symbols, start, end):
    data = yf.download(symbols, start=start, end=end)
    data.index = pd.to_datetime(data.index)
    return data['Adj Close']

def rebalance_portfolio(data, initial_investment=100000):
    monthly_data = data.resample('M').last()
    quarterly_rebalance = monthly_data.resample('Q').last()
    num_assets = len(data.columns)
    monthly_investments = initial_investment / num_assets
    
    portfolio = pd.DataFrame(index=monthly_data.index, columns=data.columns)
    investments = initial_investment  # Initial investments equally divided
    
    for date in monthly_data.index:
        if date in quarterly_rebalance.index:
            # Reconstitute: Equal investment in each asset
            investments = (portfolio.loc[date] * monthly_data.loc[date]).sum()
            monthly_investments = investments / num_assets  # Redistribute investments equally
        portfolio.loc[date] = monthly_investments / monthly_data.loc[date]
        investments = (portfolio.loc[date] * monthly_data.loc[date]).sum()

    return portfolio.multiply(monthly_data)

def top_crypto_breakdown(data):
    quarterly_data = data.resample('Q').last()
    monthly_data = data.resample('M').last()
    top_crypto_weights = pd.DataFrame()

    for quarter_end in quarterly_data.index:
        quarter_top_crypto = data.loc[:quarter_end].iloc[-1].nlargest(15)
        quarterly_weights = monthly_data.loc[quarter_end:quarter_end+pd.DateOffset(months=3), quarter_top_crypto.index].div(monthly_data.loc[quarter_end:quarter_end+pd.DateOffset(months=3), quarter_top_crypto.index].sum(axis=1), axis=0)
        top_crypto_weights = pd.concat([top_crypto_weights, quarterly_weights])

    return top_crypto_weights

st.title("Dynamic Portfolio Simulator & Crypto Analyzer")

selected_year = year

crypto = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'LTC-USD', 'EOS-USD', 'BNB-USD',
          'XTZ-USD', 'XLM-USD', 'LINK-USD', 'TRX-USD', 'NEO-USD', 'IOTA-USD', 'DASH-USD',
          'DOT-USD', 'UNI-USD', 'DOGE-USD', 'SOL-USD', 'AVAX-USD']

crypto_data = get_data(crypto, f"{selected_year}-01-01", f"{int(selected_year)+1}-01-01")
top_crypto_weights = top_crypto_breakdown(crypto_data)



st.subheader("Monthly Crypto Allocation Breakdown")
st.dataframe(top_crypto_weights.resample('M').last())
