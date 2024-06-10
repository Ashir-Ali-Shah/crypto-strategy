import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime

# Define a refined list of cryptocurrencies (to ensure a broad selection for ranking)
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
            if not data.empty:
                crypto_data[crypto] = data[['Adj Close', 'Volume']]
    return crypto_data

def calculate_portfolio_value(data, weights, initial_investment):
    daily_returns = data.pct_change().dropna()
    weighted_returns = daily_returns.dot(weights)
    cumulative_returns = (1 + weighted_returns).cumprod()
    portfolio_value = cumulative_returns * initial_investment
    return portfolio_value

def plot_monthly_allocation(prices, weights, top_cryptos):
    # Calculate monthly crypto allocation breakdown
    weighted_prices = prices.multiply(weights, axis=1)
    monthly_allocations = weighted_prices.resample('M').sum()
    top_3 = weights.nlargest(3).index.tolist()
    
    top_3_allocations = monthly_allocations[top_3]
    others_allocations = monthly_allocations.drop(columns=top_3).sum(axis=1).rename("Others")
    combined_allocations = pd.concat([top_3_allocations, others_allocations], axis=1)

    fig = px.bar(
        combined_allocations,
        x=combined_allocations.index,
        y=combined_allocations.columns,
        title='Monthly Crypto Allocation Breakdown',
        labels={'value': 'Allocation Value (USD)', 'index': 'Month'},
        template='plotly_white',
        color_discrete_sequence=['#aec6cf', '#b2df8a', '#fb9a99', '#fdbf6f']
    )

    fig.update_layout(barmode='stack', xaxis={'categoryorder': 'category ascending'})
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Cryptocurrency Portfolio Analysis")
    st.markdown("Explore different cryptocurrency portfolio strategies and their performance over a selected year.")

    with st.form("input_form"):
        st.header("User Inputs")
        col1, col2, col3 = st.columns(3)
        with col1:
            year = st.selectbox("Select Year", options=range(2023, 2019, -1), index=0)
        with col2:
            strategy = st.selectbox("Select Portfolio Strategy", options=[
                'Market Cap Weighted', 
                'Capped Market Cap Weighted', 
                'Top 15 by Volume'
            ])
        with col3:
            cap = st.number_input("Capped Market Strategy (%)", value=25, min_value=0, max_value=100)
        
        initial_investment = st.text_input("Initial Investment (USD)", value="10000")
        submit_button = st.form_submit_button("Apply Investment")

    if submit_button:
        initial_investment = float(initial_investment)

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
            excess_weights = weights[weights > cap] - cap
            weights[weights > cap] = cap
            excess_weight_sum = excess_weights.sum()

            while excess_weight_sum > 0:
                uncapped_weights = weights[weights < cap]
                if uncapped_weights.empty:
                    break
                redistribute = excess_weight_sum / uncapped_weights.size
                new_excess_weight_sum = 0
                for crypto in uncapped_weights.index:
                    if weights[crypto] + redistribute > cap:
                        new_excess_weight_sum += (weights[crypto] + redistribute) - cap
                        weights[crypto] = cap
                    else:
                        weights[crypto] += redistribute

                excess_weight_sum = new_excess_weight_sum

            weights /= weights.sum()  # Ensure the total weight sums to 1
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfolio Value', line=dict(color='#aec6cf')))
        fig.update_layout(title=f'Portfolio Value Over Time ({strategy})', xaxis_title='Date', yaxis_title='Portfolio Value (USD)')
        st.plotly_chart(fig, use_container_width=True)

        # Plot monthly allocation with top 3 colors and the rest as "Others"
        st.subheader(f'Monthly Allocation Breakdown ({strategy})')
        plot_monthly_allocation(prices.resample('M').ffill(), weights, top_cryptos)

        # Visualization of the price data of the top cryptocurrencies with improved aesthetics
        expander = st.expander("View Detailed Price Charts", expanded=False)
        with expander:
            st.markdown("Here you can explore detailed price charts for each of the top 15 cryptocurrencies:")
            col1, col2 = st.columns(2)
            for i, crypto in enumerate(top_cryptos):
                with (col1 if i % 2 == 0 else col2):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=top_cryptos_data[crypto].index, y=top_cryptos_data[crypto]['Adj Close'], mode='lines', name=f'{crypto} Adjusted Close', line=dict(color='#b2df8a')))
                    fig.update_layout(title=f'{crypto} Adjusted Close Price in {year}', xaxis_title='Date', yaxis_title='Adjusted Close Price (USD)')
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
