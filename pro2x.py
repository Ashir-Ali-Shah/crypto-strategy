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

def fetch_market_caps(cryptos):
    market_caps = {}
    for crypto in cryptos:
        try:
            ticker = yf.Ticker(crypto)
            market_cap = ticker.info.get('marketCap', 0)  # Get marketCap or default to 0
            if market_cap:  # Only add if marketCap is not zero
                market_caps[crypto] = market_cap
        except Exception as e:
            print(f"Error fetching market cap for {crypto}: {e}")
    return market_caps

def calculate_portfolio_value(data, weights, initial_investment):
    daily_returns = data.pct_change(fill_method=None).dropna()
    weighted_returns = daily_returns.dot(weights)
    cumulative_returns = (1 + weighted_returns).cumprod()
    portfolio_value = cumulative_returns * initial_investment
    
    # Calculate the number of tokens held and value based on holdings
    initial_prices = data.iloc[0]
    tokens_held = (weights * initial_investment) / initial_prices
    portfolio_values_based_on_holdings = (tokens_held * data).sum(axis=1)
    
    return portfolio_value, weighted_returns, portfolio_values_based_on_holdings

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

def market_cap_weighted(prices, market_caps, cap_percentage):
    weights = pd.Series(0, index=prices.columns)
    total_market_cap = sum(market_caps.values())
    for crypto in prices.columns:
        weights[crypto] = market_caps[crypto] / total_market_cap
    return enforce_cap(weights, cap_percentage / 100)

def capped_market_cap_weighted(prices, market_caps, cap_percentage):
    weights = market_cap_weighted(prices, market_caps, cap_percentage)
    return enforce_cap(weights, cap_percentage / 100)

def top_15_by_volume(prices):
    weights = pd.Series(0, index=prices.columns)
    total_volume = prices.sum().sum()
    for crypto in prices.columns:
        weights[crypto] = prices[crypto].sum() / total_volume
    return enforce_cap(weights, 0.25)

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
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Select Year", options=[2023, 2022, 2021], index=0)
        strategy = st.selectbox("Select Portfolio Strategy", options=[
            'Market Cap Weighted', 
            'Capped Market Cap Weighted', 
            'Top 15 by Volume'
        ])
    with col2:
        cap = st.slider("Capped Percentage (for Capped Strategy)", value=25, min_value=0, max_value=100)
        initial_investment = st.number_input("Initial Investment (USD)", value=10000, min_value=1)
    
    view_option = st.selectbox("View Option", options=['Annual', 'Quarterly'])
    if view_option == 'Quarterly':
        quarter = st.selectbox("Select Quarter", options=['Q1', 'Q2', 'Q3', 'Q4'])

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    with st.spinner('Fetching historical data...'):
        crypto_data = fetch_crypto_data(cryptos, start_date, end_date)
        if not crypto_data:
            st.error("Failed to fetch data. Please try again later.")
            return

    market_caps = fetch_market_caps(cryptos)
    
    average_volumes = {}
    price_changes = {}
    for crypto, data in crypto_data.items():
        if not data.empty:
            average_volumes[crypto] = data['Volume'].mean()
            price_changes[crypto] = ((data['Adj Close'][-1] - data['Adj Close'][0]) / data['Adj Close'][0]) * 100

    def get_quarter_dates(quarter, year):
        if quarter == 'Q1':
            return datetime(year, 1, 1), datetime(year, 3, 31)
        elif quarter == 'Q2':
            return datetime(year, 4, 1), datetime(year, 6, 30)
        elif quarter == 'Q3':
            return datetime(year, 7, 1), datetime(year, 9, 30)
        else:
            return datetime(year, 10, 1), datetime(year, 12, 31)

    if view_option == 'Quarterly':
        quarter_start, quarter_end = get_quarter_dates(quarter, year)
        top_cryptos = sorted(average_volumes, key=average_volumes.get, reverse=True)[:15]
        top_cryptos_data = {crypto: crypto_data[crypto].loc[quarter_start:quarter_end] for crypto in top_cryptos}
        prices = pd.DataFrame({crypto: data['Adj Close'] for crypto, data in top_cryptos_data.items()})
    else:
        top_cryptos = sorted(average_volumes, key=average_volumes.get, reverse=True)[:15]
        top_cryptos_data = {crypto: crypto_data[crypto] for crypto in top_cryptos}
        prices = pd.DataFrame({crypto: data['Adj Close'] for crypto, data in top_cryptos_data.items()})

    if year == 2022:
        prices = simulate_decline(prices, 2022)

    # Recalculate weights quarterly
    if view_option == 'Quarterly':
        quarter_dates = [get_quarter_dates(q, year) for q in ['Q1', 'Q2', 'Q3', 'Q4']]
        quarterly_weights = []
        for q_start, q_end in quarter_dates:
            quarter_prices = prices.loc[q_start:q_end]
            if strategy == 'Market Cap Weighted':
                weights = market_cap_weighted(quarter_prices, market_caps, cap)
            elif strategy == 'Capped Market Cap Weighted':
                weights = capped_market_cap_weighted(quarter_prices, market_caps, cap)
            else:
                weights = top_15_by_volume(quarter_prices)
            quarterly_weights.append(weights)
    else:
        if strategy == 'Market Cap Weighted':
            weights = market_cap_weighted(prices, market_caps, cap)
        elif strategy == 'Capped Market Cap Weighted':
            weights = capped_market_cap_weighted(prices, market_caps, cap)
        else:
            weights = top_15_by_volume(prices)

    view_period = f'{quarter} {year}' if view_option == 'Quarterly' else f'{year}'
    st.subheader(f'Top 15 Coins and Their Weightage ({strategy}) for {view_period}')
    top_coins_df = pd.DataFrame({
        'Coin': weights.index,
        'Weightage (%)': (weights.values * 100).round(2),
        'Latest Price (USD)': prices.iloc[-1].values.round(2),
        'Market Cap (USD)': [market_caps.get(crypto, 0) for crypto in weights.index]
    }).reset_index(drop=True)

    def style_weightage(val):
        return 'color: #90ee90; font-weight: bold;' if isinstance(val, (int, float)) else ''

    def style_usd(val):
        return 'color: #90ee90; font-weight: bold;' if isinstance(val, (int, float)) else ''

    st.dataframe(top_coins_df.style
        .applymap(style_weightage, subset=['Weightage (%)'])
        .applymap(style_usd, subset=['Latest Price (USD)', 'Market Cap (USD)'])
        .set_caption(f"Top 15 Coins and Their Weightage for {view_period}"))

    st.subheader('Top 15 Coins Weightage Visualization')
    fig, ax = plt.subplots()
    plt.style.use('dark_background')
    ax.barh(top_coins_df['Coin'], top_coins_df['Weightage (%)'], color='#90ee90', alpha=0.8)
    ax.set_xlabel('Weightage (%)')
    ax.set_title(f'Top 15 Coins and Their Weightage for {view_period}')
    st.pyplot(fig)

    monthly_prices = prices.resample('M').ffill()

    st.subheader(f'Portfolio Value Over Time ({strategy}) for {view_period}')
    portfolio_value, weighted_returns, portfolio_values_based_on_holdings = calculate_portfolio_value(prices, weights, initial_investment)
    allocation_percentage = portfolio_value / initial_investment * 100
    holdings_allocation_percentage = portfolio_values_based_on_holdings / initial_investment * 100
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(allocation_percentage.index, allocation_percentage, label='Portfolio Allocation (%)', color='#90ee90', alpha=0.8)
    ax.plot(holdings_allocation_percentage.index, holdings_allocation_percentage, label='Holdings Allocation (%)', color='#1f77b4', alpha=0.8)
    ax.set_title(f'Portfolio Allocation Over Time ({strategy}) for {view_period}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Allocation (%)')
    ax.set_xticks(allocation_percentage.index[::30])
    ax.set_xticklabels(allocation_percentage.index.strftime('%b %Y')[::30], rotation=90)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    st.subheader(f'Monthly Prices for Top 15 Coins ({strategy}) for {view_period}')
    monthly_prices_styled = monthly_prices.style.format("{:.2f}").applymap(lambda x: 'color: #90ee90;' if x >= 0 else 'color: #ff4d4d;')
    st.dataframe(monthly_prices_styled.set_caption(f"Monthly Prices for {view_period}"))

    st.subheader('Monthly Prices Visualization')
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.style.use('dark_background')
    for crypto in top_cryptos:
        ax.plot(monthly_prices.index, monthly_prices[crypto], label=crypto)
    ax.set_title(f'Monthly Prices for Top 15 Coins for {view_period}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # Preparing the data for the monthly allocation breakdown with bars for each month in the quarter or the year
    def prepare_monthly_allocation(prices, weights, initial_investment):
        monthly_returns = prices.pct_change().dropna()
        cumulative_returns = (monthly_returns + 1).cumprod()
        weighted_cumulative_returns = cumulative_returns.mul(weights, axis=1)
        monthly_allocations = weighted_cumulative_returns.mul(initial_investment).resample('M').last()
        return monthly_allocations

    monthly_allocations = prepare_monthly_allocation(prices, weights, initial_investment)

    st.subheader(f'Monthly Allocation Breakdown ({strategy}) for {view_period}')
    fig, ax = plt.subplots(figsize=(14, 7))
    monthly_allocations.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
    ax.set_title(f'Monthly Crypto Allocation Breakdown for {view_period}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value in USD')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

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
