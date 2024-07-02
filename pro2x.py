import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Define an extended list of cryptocurrencies
cryptos = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'LTC-USD', 'EOS-USD', 'BNB-USD',
    'XTZ-USD', 'XLM-USD', 'LINK-USD', 'TRX-USD', 'NEO-USD', 'IOTA-USD', 'DASH-USD',
    'DOT-USD', 'UNI-USD', 'DOGE-USD', 'SOL-USD', 'AVAX-USD', 'FIL-USD', 'AAVE-USD',
    'ALGO-USD', 'ATOM-USD', 'VET-USD', 'ICP-USD', 'FTT-USD', 'SAND-USD', 'AXS-USD',
    'MATIC-USD', 'THETA-USD', 'EGLD-USD', 'KSM-USD', 'CAKE-USD', 'MKR-USD',
    'COMP-USD', 'ZEC-USD', 'XMR-USD', 'KCS-USD', 'HT-USD', 'OKB-USD', 'LEO-USD',
    'WAVES-USD', 'MIOTA-USD', 'NEAR-USD', 'APE-USD', 'GMT-USD', 'GRT-USD',
    'ENJ-USD', 'MANA-USD', 'GALA-USD', 'CHZ-USD', 'FLOW-USD', 'SUSHI-USD', 'YFI-USD',
    'CRV-USD', '1INCH-USD', 'SNX-USD', 'CELO-USD', 'BAT-USD', 'QTUM-USD',
    'ZIL-USD', 'SC-USD', 'DCR-USD', 'XEM-USD', 'LSK-USD', 'RVN-USD', 'KDA-USD', 'OMG-USD',
    'NEXO-USD', 'HNT-USD', 'ZRX-USD', 'STX-USD', 'UST-USD', 'PAXG-USD', 'TFUEL-USD',
    'ANKR-USD', 'REN-USD', 'ICX-USD', 'FTM-USD', 'SRM-USD', 'CVC-USD', 'ALPHA-USD',
    'AUDIO-USD', 'CKB-USD', 'BNT-USD', 'LPT-USD', 'WAXP-USD', 'SXP-USD', 'OCEAN-USD',
    'RLY-USD', 'SKL-USD', 'UMA-USD', 'ONT-USD', 'RAY-USD', 'RSR-USD', 'AMPL-USD', 'ILV-USD'
]

@st.cache_data(show_spinner=False)
def fetch_crypto_data(cryptos, start, end):
    crypto_data = {}
    for crypto in cryptos:
        data = yf.download(crypto, start=start, end=end)
        if not data.empty:
            crypto_data[crypto] = data[['Adj Close', 'Volume']]
    return crypto_data

def calculate_portfolio_value(data, weights, initial_investment):
    daily_returns = data.pct_change(fill_method=None).dropna()
    weighted_returns = daily_returns.dot(weights)
    cumulative_returns = (1 + weighted_returns).cumprod()
    portfolio_value = cumulative_returns * initial_investment
    return portfolio_value, weighted_returns

def plot_monthly_allocation(monthly_prices, weights, top_3):
    plt.style.use('dark_background')
    monthly_returns = monthly_prices.pct_change(fill_method=None).dropna()
    monthly_allocations = (monthly_returns + 1).cumprod() * weights * monthly_prices.iloc[0]
    top_3_allocations = monthly_allocations[top_3]
    others_allocations = monthly_allocations.drop(columns=top_3).sum(axis=1).rename("Others")
    combined_allocations = pd.concat([top_3_allocations, others_allocations], axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    combined_allocations.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
    ax.set_title('Monthly Crypto Allocation Breakdown')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value in USD')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

def simulate_decline(data, year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    decline_factor = np.linspace(1, 0.5, len(date_range))
    decline = pd.Series(index=date_range, data=decline_factor)
    simulated_data = data.copy()
    for date in date_range:
        if date in simulated_data.index:
            simulated_data.loc[date] = simulated_data.loc[date] * decline.loc[date]
    return simulated_data

def enforce_cap(weights, cap):
    while weights.max() > cap:
        excess = weights[weights > cap] - cap
        total_excess = excess.sum()
        adjusted_weights = weights.copy()
        adjusted_weights[weights > cap] = cap
        remaining_cryptos = adjusted_weights[adjusted_weights < cap]
        adjusted_weights[adjusted_weights < cap] += total_excess * (remaining_cryptos / remaining_cryptos.sum())
        weights = adjusted_weights / adjusted_weights.sum()
    return weights

def market_cap_weighted(prices):
    weights = pd.Series(0, index=prices.columns)
    weights['BTC-USD'] = 0.25
    remaining_cryptos = prices.columns.drop('BTC-USD')
    remaining_weights = (prices.iloc[-1][remaining_cryptos] / prices.iloc[-1][remaining_cryptos].sum()) * 0.75
    weights[remaining_cryptos] = remaining_weights
    return enforce_cap(weights, 0.25)

def capped_market_cap_weighted(prices, cap_percentage):
    weights = market_cap_weighted(prices)
    return enforce_cap(weights, cap_percentage / 100)

def top_15_by_volume(prices):
    weights = pd.Series(0, index=prices.columns)
    weights['BTC-USD'] = 0.25
    remaining_weight = 0.75
    other_cryptos = prices.columns.drop('BTC-USD')
    equal_weight = remaining_weight / len(other_cryptos)
    weights[other_cryptos] = equal_weight
    return enforce_cap(weights, 0.25)

def main():
    st.markdown("""
        <style>
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        .fade-in {
            animation: fadeIn 2s;
        }
        .center-text {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='center-text fade-in'>Crypto Portfolio Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='center-text fade-in'>Analyze and optimize your cryptocurrency portfolio with various strategies based on trading volumes and market capitalization.</h2>", unsafe_allow_html=True)
    st.markdown("<div class='center-text'><h3 class='fade-in'>User Inputs</h3></div>", unsafe_allow_html=True)
    year = st.selectbox("Select Year", options=[2023, 2022, 2021], index=0)
    strategy = st.selectbox("Select Portfolio Strategy", options=[
        'Market Cap Weighted', 
        'Capped Market Cap Weighted', 
        'Top 15 by Volume'
    ])
    cap = st.number_input("Capped Percentage (for Capped Strategy)", value=25, min_value=0, max_value=100)
    initial_investment = st.number_input("Initial Investment (USD)", value=10000, min_value=1)

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    with st.spinner('Fetching historical data...'):
        crypto_data = fetch_crypto_data(cryptos, start_date, end_date)
        if not crypto_data:
            st.error("Failed to fetch data. Please try again later.")
            return

    average_volumes = {}
    price_changes = {}
    for crypto, data in crypto_data.items():
        if not data.empty:
            average_volumes[crypto] = data['Volume'].mean()
            price_changes[crypto] = ((data['Adj Close'][-1] - data['Adj Close'][0]) / data['Adj Close'][0]) * 100

    top_cryptos = sorted(average_volumes, key=average_volumes.get, reverse=True)[:15]
    top_cryptos_data = {crypto: crypto_data[crypto] for crypto in top_cryptos}
    prices = pd.DataFrame({crypto: data['Adj Close'] for crypto, data in top_cryptos_data.items()})

    if year == 2022:
        prices = simulate_decline(prices, 2022)

    if strategy == 'Market Cap Weighted':
        weights = market_cap_weighted(prices)
    elif strategy == 'Capped Market Cap Weighted':
        weights = capped_market_cap_weighted(prices, cap)
    else:
        weights = top_15_by_volume(prices)

    st.subheader(f'Top 15 Coins and Their Weightage ({strategy})')
    top_coins_df = pd.DataFrame({
        'Coin': weights.index,
        'Weightage (%)': (weights.values * 100).round(2),
        'Latest Price (USD)': prices.iloc[-1].values.round(2)
    }).reset_index(drop=True)

    def style_weightage(val):
        return 'color: #90ee90; font-weight: bold;' if isinstance(val, (int, float)) else ''

    def style_usd(val):
        return 'color: #90ee90; font-weight: bold;' if isinstance(val, (int, float)) else ''

    st.dataframe(top_coins_df.style
        .applymap(style_weightage, subset=['Weightage (%)'])
        .applymap(style_usd, subset=['Latest Price (USD)'])
        .set_caption("Top 15 Coins and Their Weightage"))

    st.subheader('Top 15 Coins Weightage Visualization')
    fig, ax = plt.subplots()
    plt.style.use('dark_background')
    ax.barh(top_coins_df['Coin'], top_coins_df['Weightage (%)'], color='#90ee90', alpha=0.8)
    ax.set_xlabel('Weightage (%)')
    ax.set_title('Top 15 Coins and Their Weightage')
    st.pyplot(fig)

    monthly_prices = prices.resample('ME').ffill()
    st.subheader(f'Monthly Prices for Top 15 Coins ({strategy})')
    monthly_prices_styled = monthly_prices.style.format("{:.2f}").applymap(lambda x: 'color: #90ee90;' if x >= 0 else 'color: #ff4d4d;')
    st.dataframe(monthly_prices_styled.set_caption("Monthly Prices"))

    st.subheader('Monthly Prices Visualization')
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.style.use('dark_background')
    for crypto in top_cryptos:
        ax.plot(monthly_prices.index, monthly_prices[crypto], label=crypto)
    ax.set_title('Monthly Prices for Top 15 Coins')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    portfolio_value, weighted_returns = calculate_portfolio_value(prices, weights, initial_investment)

    st.subheader(f'Portfolio Allocation Over Time ({strategy})')
    allocation_percentage = portfolio_value / initial_investment * 100
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(allocation_percentage.index, allocation_percentage, label='Portfolio Allocation (%)', color='#90ee90', alpha=0.8)
    ax.set_title(f'Portfolio Allocation Over Time ({strategy})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Allocation (%)')
    ax.set_xticks(allocation_percentage.index[::30])
    ax.set_xticklabels(allocation_percentage.index.strftime('%b %Y')[::30], rotation=90)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    st.subheader(f'Monthly Allocation Breakdown ({strategy})')
    plot_monthly_allocation(monthly_prices, weights, top_cryptos[:3])

    expander = st.expander("View Detailed Price Charts")
    with expander:
        st.markdown("Here you can explore detailed price charts for each of the top 15 cryptocurrencies:")
        col1, col2 = st.columns(2)
        for i, crypto in enumerate(top_cryptos):
            with (col1 if i % 2 == 0 else col2):
                fig, ax = plt.subplots()
                plt.style.use('dark_background')
                ax.plot(top_cryptos_data[crypto]['Adj Close'], label=f'{crypto} Adjusted Close', color='#add8e6', alpha=0.8)
                ax.set_title(f"{crypto} Adjusted Close Price in {year}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Adjusted Close Price (USD)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
