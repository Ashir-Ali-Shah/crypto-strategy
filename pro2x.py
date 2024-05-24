import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

# Fetch historical data for the ETFs
def fetch_data(etfs, start_date, end_date):
    data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    if isinstance(data, pd.Series):  # Convert to DataFrame if a single ETF
        data = data.to_frame()
    data.index = pd.to_datetime(data.index)  # Ensure the index is a DatetimeIndex
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

# Fetch data and calculate top 15 cryptos by market cap for the given year
def get_top_cryptos_by_year(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    data = fetch_data(etf_list, start_date, end_date)
    end_of_year_data = data.iloc[-1]
    top_15_cryptos = end_of_year_data.nlargest(15).index.tolist()
    return top_15_cryptos

def fetch_crypto_data(cryptos, start, end):
    crypto_data = {}
    for crypto in cryptos:
        data = yf.download(crypto, start=start, end=end)
        crypto_data[crypto] = data[['Adj Close', 'Volume']]
    return crypto_data
def main():
    st.title('Crypto Portfolio Analysis')
    
    st.sidebar.header('User Input')
    selected_year = st.sidebar.selectbox('Select Year', options=range(2020, 2024))
    initial_investment = st.sidebar.number_input('Initial Investment', value=10000)
    
    top_15_cryptos = get_top_cryptos_by_year(selected_year)
    
    data = fetch_data(top_15_cryptos, f'{selected_year}-01-01', f'{selected_year}-12-31')
    portfolio_value = calculate_portfolio_growth(data, initial_investment)
    plot_portfolio_growth(portfolio_value)
    
    monthly_allocations = calculate_monthly_allocation(data)
    plot_monthly_allocation(monthly_allocations)
    # Fetch historical data
    cryptos = etf_list
    st.title("Top Cryptocurrencies by Trading Volume")
    st.markdown("Explore the top 15 cryptocurrencies based on their trading volumes and year-over-year performance.")

    # Sidebar for input to keep main area less cluttered
    year = selected_year

    # Define the start and end dates for the given year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    # Fetch historical data
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
    st.subheader(f"Top 15 Cryptocurrencies by Trading Volume in {year}")
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
                ax.plot(crypto_data[crypto]['Adj Close'], label=f'{crypto} Adjusted Close', color='lightgreen', alpha=0.7)
                ax.set_title(f"{crypto} Adjusted Close Price in {year}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Adjusted Close Price (USD)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
    
if __name__ == "__main__":
    main()

