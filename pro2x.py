import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np



# Define the list of ETF tokens
etf_list = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'LTC-USD', 'EOS-USD', 'BNB-USD',
    'XTZ-USD', 'XLM-USD', 'LINK-USD', 'TRX-USD', 'NEO-USD', 'IOTA-USD', 'DASH-USD',
    'DOT-USD', 'UNI-USD', 'DOGE-USD', 'SOL-USD', 'AVAX-USD', 'FIL-USD', 'AAVE-USD',
    'ALGO-USD', 'ATOM-USD', 'VET-USD', 'ICP-USD', 'FTT-USD', 'SAND-USD', 'AXS-USD',
    'MATIC-USD', 'THETA-USD', 'EGLD-USD', 'KSM-USD', 'CAKE-USD', 'MKR-USD',
    'COMP-USD', 'ZEC-USD', 'XMR-USD', 'KCS-USD', 'HT-USD', 'OKB-USD', 'LEO-USD',
    'WAVES-USD', 'MIOTA-USD', 'LUNA1-USD', 'NEAR-USD', 'APE-USD', 'GMT-USD', 'GRT-USD',
    'ENJ-USD', 'MANA-USD', 'GALA-USD', 'CHZ-USD', 'FLOW-USD', 'SUSHI-USD', 'YFI-USD',
    'CRV-USD', '1INCH-USD', 'SNX-USD', 'CELO-USD', 'AAVE-USD', 'BAT-USD', 'QTUM-USD',
    'ZIL-USD', 'SC-USD', 'DCR-USD', 'XEM-USD', 'LSK-USD', 'RVN-USD', 'KDA-USD', 'OMG-USD',
    'NEXO-USD', 'HNT-USD', 'ZRX-USD', 'STX-USD', 'UST-USD', 'PAXG-USD', 'TFUEL-USD',
    'ANKR-USD', 'REN-USD', 'ICX-USD', 'FTM-USD', 'SRM-USD', 'CVC-USD', 'ALPHA-USD',
    'AUDIO-USD', 'CKB-USD', 'BNT-USD', 'LPT-USD', 'WAXP-USD', 'SXP-USD', 'OCEAN-USD',
    'RLY-USD', 'SKL-USD', 'UMA-USD', 'ONT-USD', 'RAY-USD', 'RSR-USD', 'AMPL-USD', 'ILV-USD'
]
x = 'BTC-USD'

# Fetch historical data for the ETFs
def fetch_data(etfs, start_date='2020-01-01', end_date='2024-01-01'):
    data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    if isinstance(data, pd.Series):  # Convert to DataFrame if a single ETF
        data = data.to_frame()
    return data

# Calculate portfolio value over time
def calculate_portfolio_growth(data, initial_investment):
    daily_returns = data.pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod()
    portfolio_value = cumulative_returns * (initial_investment / len(data.columns))
    portfolio_value['Total'] = portfolio_value.sum(axis=1)
    return portfolio_value

# Calculate monthly crypto allocation breakdown
def calculate_monthly_allocation(data):
    monthly_data = data.resample('M').ffill()
    monthly_allocations = monthly_data.div(monthly_data.sum(axis=1), axis=0) * 100
    return monthly_allocations

# Plot the portfolio growth
def plot_portfolio_growth(portfolio_value):
    plt.figure(figsize=(14, 7))
    for column in portfolio_value.columns:
        plt.plot(portfolio_value.index, portfolio_value[column], label=column)
    plt.title('Portfolio Growth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Plot the monthly crypto allocation breakdown
def plot_monthly_allocation(monthly_allocations):
    plt.figure(figsize=(14, 7))
    monthly_allocations.plot(kind='bar', stacked=True, figsize=(14, 7))
    plt.title('Monthly Crypto Allocation Breakdown')
    plt.xlabel('Month')
    plt.ylabel('Allocation Percentage')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    st.pyplot(plt)

# Streamlit app
def main():
    st.title('Crypto Portfolio Analysis')
    
    st.sidebar.header('User Input')
    etfs = st.sidebar.text_input('Enter ETF symbols separated by commas', 'BTC-USD')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2024-01-01'))
    initial_investment = st.sidebar.number_input('Initial Investment', value=10000)
    etf_list_input = [etf.strip() for etf in etfs.split(',')]
    
    data = fetch_data(etf_list_input, start_date=start_date, end_date=end_date)
    portfolio_value = calculate_portfolio_growth(data, initial_investment)
    plot_portfolio_growth(portfolio_value)
    
    monthly_allocations = calculate_monthly_allocation(data)
    plot_monthly_allocation(monthly_allocations)

if __name__ == "__main__":
    main()



# Set page config

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
    'MATIC-USD', 'THETA-USD', 'EGLD-USD', 'KSM-USD', 'CAKE-USD', 'MKR-USD',
    'COMP-USD', 'ZEC-USD', 'XMR-USD', 'KCS-USD', 'HT-USD', 'OKB-USD', 'LEO-USD',
    'WAVES-USD', 'MIOTA-USD', 'LUNA1-USD', 'NEAR-USD', 'APE-USD', 'GMT-USD', 'GRT-USD',
    'ENJ-USD', 'MANA-USD', 'GALA-USD', 'CHZ-USD', 'FLOW-USD', 'SUSHI-USD', 'YFI-USD',
    'CRV-USD', '1INCH-USD', 'SNX-USD', 'CELO-USD', 'AAVE-USD', 'BAT-USD', 'QTUM-USD',
    'ZIL-USD', 'SC-USD', 'DCR-USD', 'XEM-USD', 'LSK-USD', 'RVN-USD', 'KDA-USD', 'OMG-USD',
    'NEXO-USD', 'HNT-USD', 'ZRX-USD', 'STX-USD', 'UST-USD', 'PAXG-USD', 'TFUEL-USD',
    'ANKR-USD', 'REN-USD', 'ICX-USD', 'FTM-USD', 'SRM-USD', 'CVC-USD', 'ALPHA-USD',
    'AUDIO-USD', 'CKB-USD', 'BNT-USD', 'LPT-USD', 'WAXP-USD', 'SXP-USD', 'OCEAN-USD',
    'RLY-USD', 'SKL-USD', 'UMA-USD', 'ONT-USD', 'RAY-USD', 'RSR-USD', 'AMPL-USD', 'ILV-USD'
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
    # Resample data monthly
    monthly_data = data.resample('M').last()
    
    # Ensure every month within the data range is included in monthly_data
    all_months = pd.date_range(start=data.index.min(), end=data.index.max(), freq='M')
    monthly_data = monthly_data.reindex(all_months, method='ffill')  # Forward fill to handle missing data

    top_crypto_weights = pd.DataFrame()

    for month_end in monthly_data.index:
        # Select the top 15 cryptocurrencies by their last available price of the month
        if not monthly_data.loc[month_end].isna().all():  # Ensure there's valid data for the month
            month_top_crypto = monthly_data.loc[month_end].nlargest(15).index

            # Calculate weights for the top cryptos for the current month
            weights = monthly_data.loc[month_end, month_top_crypto] / monthly_data.loc[month_end, month_top_crypto].sum()
            top_crypto_weights = pd.concat([top_crypto_weights, pd.DataFrame([weights], index=[month_end])])

    return top_crypto_weights
