import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
import numpy as np

# Function to calculate Intraday Momentum Index (IMI)
def calculate_imi(data, length=90):
    price_data = (data['High'] + data['Low']) / 2  # Average price
    up_moves = []
    down_moves = []

    for i in range(1, len(price_data)):
        avg_price_today = price_data[i]
        avg_price_yesterday = price_data[i - 1]

        if avg_price_today > avg_price_yesterday:
            up_moves.append(avg_price_today - avg_price_yesterday)
            down_moves.append(0)
        else:
            down_moves.append(avg_price_yesterday - avg_price_today)
            up_moves.append(0)

    up_moves = pd.Series(up_moves)
    down_moves = pd.Series(down_moves)

    avg_up = up_moves.rolling(window=length).mean()
    avg_down = down_moves.rolling(window=length).mean()

    imi = 100 * (avg_up / (avg_up + avg_down))
    
    return imi

# Function to plot option indicators
def plot_options_indicators(data, ticker):
    fig, axs = plt.subplots(4, 1, figsize=(16, 18))

    # Plotting Relative Strength Index (RSI)
    axs[0].plot(data['Date'], data['RSI'], label='RSI', color='purple')
    axs[0].axhline(70, linestyle='--', color='red', label='Overbought')
    axs[0].axhline(30, linestyle='--', color='green', label='Oversold')
    axs[0].set_title(f'{ticker} - Relative Strength Index (RSI)')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('RSI')
    axs[0].legend()

    # Plotting Bollinger Bands
    axs[1].plot(data['Date'], data['Close'], label='Close Price', color='black')
    axs[1].plot(data['Date'], data['Bollinger_Upper'], label='Upper Band', color='blue')
    axs[1].plot(data['Date'], data['Bollinger_Lower'], label='Lower Band', color='orange')
    axs[1].fill_between(data['Date'], data['Bollinger_Lower'], data['Bollinger_Upper'], color='gray', alpha=0.3)
    axs[1].set_title(f'{ticker} - Bollinger Bands')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].legend()

    # Plotting Intraday Momentum Index (IMI) and Money Flow Index (MFI)
    axs[2].plot(data['Date'], data['IMI'], label='Intraday Momentum Index (IMI)', color='blue')
    axs[2].axhline(70, linestyle='--', color='red', label='IMI Overbought')
    axs[2].axhline(30, linestyle='--', color='green', label='IMI Oversold')
    
    axs[2].plot(data['Date'], data['MFI'], label='Money Flow Index (MFI)', color='orange')
    axs[2].axhline(80, linestyle='--', color='purple', label='MFI Overbought')
    axs[2].axhline(20, linestyle='--', color='brown', label='MFI Oversold')
    
    axs[2].set_title(f'{ticker} - IMI and MFI')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Index Value')
    axs[2].legend()

    # Plotting Put-Call Ratio (PCR)
    axs[3].plot(data['Date'], data['PCR'], label='Put-Call Ratio (PCR)', color='magenta')
    axs[3].set_title(f'{ticker} - Put-Call Ratio (PCR)')
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('PCR')
    axs[3].legend()

    plt.tight_layout()
    st.pyplot(fig)

st.title("Options Indicators Dashboard")

# User input for stock tickers
tickers_input = st.text_input("Enter stock tickers separated by commas (e.g., TSLA,NVDA,AMZN):", "TSLA,NVDA,AMZN")
tickers = [ticker.strip() for ticker in tickers_input.split(",")]

for ticker in tickers:
    st.write(f"Fetching options data for {ticker}...")
    stock = yf.Ticker(ticker)

    # Get expiration dates for the options
    expiration_dates = stock.options
    
    # Collect option data for each expiration date
    option_list = []
    
    for expiration in expiration_dates:
        options = stock.option_chain(expiration)
        calls = options.calls
        puts = options.puts
        
        # Add relevant data to the option list
        for index, row in calls.iterrows():
            option_list.append({
                'Ticker': ticker,
                'Type': 'Call',
                'Expiration': expiration,
                'Strike': row['strike'],
                'Volume': row['volume'],
                'Open Interest': row['openInterest'],
                'Implied Volatility': row['impliedVolatility'],
                'Close': row['lastPrice'],
                'High': 0,  # Placeholder since we don't have highPrice
                'Low': 0,   # Placeholder since we don't have lowPrice
                'Date': pd.to_datetime(expiration)  # Use the expiration date directly
            })
        
        for index, row in puts.iterrows():
            option_list.append({
                'Ticker': ticker,
                'Type': 'Put',
                'Expiration': expiration,
                'Strike': row['strike'],
                'Volume': row['volume'],
                'Open Interest': row['openInterest'],
                'Implied Volatility': row['impliedVolatility'],
                'Close': row['lastPrice'],
                'High': 0,  # Placeholder since we don't have highPrice
                'Low': 0,   # Placeholder since we don't have lowPrice
                'Date': pd.to_datetime(expiration)  # Use the expiration date directly
            })

    # Convert option list to DataFrame
    option_df = pd.DataFrame(option_list)

    # Use historical prices for IMI and MFI calculation
    historical_data = stock.history(period="1mo")  # Get historical price data for the underlying stock
    historical_data.reset_index(inplace=True)  # Reset index to get 'Date' as a column

    # Ensure that the necessary columns are in float format
    historical_data['High'] = historical_data['High'].astype(float)
    historical_data['Low'] = historical_data['Low'].astype(float)
    historical_data['Close'] = historical_data['Close'].astype(float)
    historical_data['Volume'] = historical_data['Volume'].astype(float)

    # Calculate IMI using historical data
    option_df['IMI'] = calculate_imi(historical_data)

    # Calculate MFI using the underlying stock's data
    historical_data['MFI'] = ta.mfi(
        high=historical_data['High'], 
        low=historical_data['Low'], 
        close=historical_data['Close'], 
        volume=historical_data['Volume'], 
        length=14
    )

    # Remove timezone information from historical_data['Date']
    historical_data['Date'] = historical_data['Date'].dt.tz_localize(None)

    # Combine MFI with option_df based on the corresponding dates
    option_df = option_df.merge(historical_data[['Date', 'MFI']], on='Date', how='left')

    # Calculate additional indicators
    # 1. Relative Strength Index (RSI)
    option_df['RSI'] = ta.rsi(option_df['Volume'], length=14)

    # 2. Bollinger Bands
    option_df['Bollinger_MA'] = option_df['Volume'].rolling(window=20).mean()
    option_df['Bollinger_STD'] = option_df['Volume'].rolling(window=20).std()
    option_df['Bollinger_Upper'] = option_df['Bollinger_MA'] + (option_df['Bollinger_STD'] * 2)
    option_df['Bollinger_Lower'] = option_df['Bollinger_MA'] - (option_df['Bollinger_STD'] * 2)

    # 5. Calculate Put-Call Ratio (PCR)
    total_calls = option_df[option_df['Type'] == 'Call']['Volume'].sum()
    total_puts = option_df[option_df['Type'] == 'Put']['Volume'].sum()
    option_df['PCR'] = total_puts / total_calls if total_calls != 0 else 0

    # Sort by volume in decreasing order and select the top 20 options
    top_20_options = option_df.sort_values(by='Volume', ascending=False).head(20)

    # Create two columns for the dashboard
    col1, col2 = st.columns([2, 1])

    # Left column: Plot the option indicators and display trading indicator metrics
    with col1:
        st.subheader(f"{ticker} - Top 20 Options by Volume")
        st.dataframe(top_20_options[['Type', 'Strike', 'Expiration', 'Volume', 'Open Interest', 'Implied Volatility']])
        plot_options_indicators(option_df, ticker)

    # Right column: Show the trading indicators as metrics
    with col2:
        st.subheader(f"{ticker} Trading Indicators")
        st.metric("RSI", option_df['RSI'].iloc[-1] if not option_df['RSI'].isnull().all() else "N/A")
        st.metric("IMI", option_df['IMI'].iloc[-1] if not option_df['IMI'].isnull().all() else "N/A")
        st.metric("MFI", option_df['MFI'].iloc[-1] if not option_df['MFI'].isnull().all() else "N/A")
        st.metric("Put-Call Ratio (PCR)", option_df['PCR'].iloc[-1] if not option_df['PCR'].isnull().all() else "N/A")
