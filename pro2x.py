import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px;
    }
    .stSidebar .stSelectbox, .stSidebar .stNumberInput {
        background-color: #ffffff;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

def calculate_portfolio_value(data, weights, initial_investment):
    daily_returns = data.pct_change().dropna()
    weighted_returns = daily_returns.dot(weights)
    cumulative_returns = (1 + weighted_returns).cumprod()
    portfolio_value = cumulative_returns * initial_investment
    return portfolio_value

def plot_monthly_allocation(monthly_allocations, top_3):
    plt.figure(figsize=(14, 7))
    top_3_allocations = monthly_allocations[top_3]
    others_allocations = monthly_allocations.drop(columns=top_3).sum(axis=1).rename("Others")
    combined_allocations = pd.concat([top_3_allocations, others_allocations], axis=1)
    
    combined_allocations.plot(kind='bar', stacked=True, figsize=(14, 7), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    plt.title('Monthly Crypto Allocation Breakdown')
    plt.xlabel('Month')
    plt.ylabel('Allocation Percentage')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(plt)

def main():
    st.title("Top Cryptocurrencies by Trading Volume")
    st.markdown("Explore the top 15 cryptocurrencies based on their trading volumes and year-over-year performance.")

    # Sidebar for input to keep main area less cluttered
    with st.sidebar:
        st.header("User Inputs")
        year = st.selectbox("Select Year", options=range(2023, 2019, -1), index=0)
        strategy = st.selectbox("Select Portfolio Strategy", options=[
            'Market Cap Weighted', 
            'Capped Market Cap Weighted', 
            'Top 15 by Volume'
        ])
        cap = st.number_input("Capped Percentage (for Capped Strategy)", value=25, min_value=0, max_value=100)
        initial_investment = st.number_input("Initial Investment (USD)", value=10000, min_value=1)

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

    # Get data for top 15 cryptos
    top_cryptos_data = {crypto: crypto_data[crypto] for crypto in top_cryptos}

    # Prepare data for portfolio calculations
    prices = pd.DataFrame({crypto: data['Adj Close'] for crypto, data in top_cryptos_data.items()})

    # Define portfolio strategies
    def market_cap_weighted(prices):
        latest_prices = prices.iloc[-1]
        total_market_cap = latest_prices.sum()
        weights = latest_prices / total_market_cap
        return weights

    def capped_market_cap_weighted(prices, cap_percentage):
        weights = market_cap_weighted(prices)
        cap = cap_percentage / 100
        weights = weights.clip(upper=cap)
        weights /= weights.sum()
        return weights

    def top_15_by_volume(prices):
        return pd.Series([1/len(prices.columns)] * len(prices.columns), index=prices.columns)

    # Determine portfolio weights based on the selected strategy
    if strategy == 'Market Cap Weighted':
        weights = market_cap_weighted(prices)
    elif strategy == 'Capped Market Cap Weighted':
        weights = capped_market_cap_weighted(prices, cap)
    else:
        weights = top_15_by_volume(prices)

    # Calculate portfolio value over time
    portfolio_value = calculate_portfolio_value(prices, weights, initial_investment)

    # Plot portfolio value
    st.subheader(f'Portfolio Value Over Time ({strategy})')
    fig, ax = plt.subplots()
    ax.plot(portfolio_value, label='Portfolio Value', color='#2ca02c', alpha=0.8)
    ax.set_title(f'Portfolio Value Over Time ({strategy})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (USD)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # Calculate monthly crypto allocation breakdown
    monthly_allocations = prices.resample('M').ffill().pct_change().dropna().dot(weights)

    # Plot monthly allocation with top 3 colors and the rest as "Others"
    st.subheader(f'Monthly Allocation Breakdown ({strategy})')
    plot_monthly_allocation(prices.resample('M').ffill(), top_cryptos[:3])

    # Visualization of the price data of the top cryptocurrencies with improved aesthetics
    expander = st.expander("View Detailed Price Charts")
    with expander:
        st.markdown("Here you can explore detailed price charts for each of the top 15 cryptocurrencies:")
        col1, col2 = st.columns(2)
        for i, crypto in enumerate(top_cryptos):
            with (col1 if i % 2 == 0 else col2):
                fig, ax = plt.subplots()
                ax.plot(top_cryptos_data[crypto]['Adj Close'], label=f'{crypto} Adjusted Close', color='#1f77b4', alpha=0.8)
                ax.set_title(f"{crypto} Adjusted Close Price in {year}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Adjusted Close Price (USD)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
