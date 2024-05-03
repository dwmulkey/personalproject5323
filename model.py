# %%
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys

# Set up matplotlib to work without an interactive backend
import matplotlib
matplotlib.use('Agg')

def download_data(ticker):
    stock_data = yf.download(ticker, period="5y")
    index_data = yf.download('^GSPC', period="5y")  # S&P 500 as a benchmark
    return stock_data, index_data

def clean_and_prepare_data(stock_data, index_data):
    stock_data = stock_data.loc[index_data.index.intersection(stock_data.index)]
    index_data = index_data.loc[stock_data.index]
    stock_returns = stock_data['Close'].pct_change().dropna()
    index_returns = index_data['Close'].pct_change().dropna()
    combined_data = pd.DataFrame({
        'Stock_Returns': stock_returns,
        'Index_Returns': index_returns
    }).dropna()
    return combined_data

def calculate_volatility(data):
    data['Volatility'] = data['Stock_Returns'].rolling(window=252).std() * np.sqrt(252)
    return data.dropna()

def calculate_sharpe_ratio(stock_returns, risk_free_rate=0.0467):
    daily_risk_free_rate = risk_free_rate / 252
    excess_returns = stock_returns - daily_risk_free_rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_sortino_ratio(stock_returns, risk_free_rate=0.0467):
    daily_risk_free_rate = risk_free_rate / 252
    negative_returns = stock_returns[stock_returns < 0]
    downside_std = negative_returns.std()
    return (stock_returns.mean() - daily_risk_free_rate) / downside_std * np.sqrt(252)

def calculate_beta(stock_returns, index_returns):
    covariance = np.cov(stock_returns, index_returns)
    beta = covariance[0, 1] / covariance[1, 1]
    return beta

def calculate_alpha(stock_returns, index_returns, risk_free_rate=0.0467, beta=None):
    if beta is None:
        beta = calculate_beta(stock_returns, index_returns)
    daily_risk_free_rate = risk_free_rate / 252
    stock_excess_returns = stock_returns.mean() - daily_risk_free_rate
    index_excess_returns = index_returns.mean() - daily_risk_free_rate
    alpha = (stock_excess_returns - beta * index_excess_returns) * 252
    return alpha

def value_at_risk(position, c, sigma):
    var = position * norm.ppf(1 - c) * sigma
    return abs(var)

def expected_shortfall(position, c, sigma):
    ES = position * sigma * norm.pdf(norm.ppf(1 - c)) / c
    return abs(ES)

def calculate_macd(stock_prices, slow=26, fast=12, signal=9):
    exp1 = stock_prices.ewm(span=fast, adjust=False).mean()
    exp2 = stock_prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_rsi(stock_prices, periods=14):
    delta = stock_prices.diff()
    gains = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    losses = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def calculate_sma(stock_prices, window=50):
    return stock_prices.rolling(window=window).mean()

import matplotlib.colors as mcolors

def plot_stock_data(stock_prices):
    macd, signal_line = calculate_macd(stock_prices)
    rsi = calculate_rsi(stock_prices)
    sma50 = calculate_sma(stock_prices, 50)
    sma200 = calculate_sma(stock_prices, 200)

    # Set high-resolution DPI for the output
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['font.family'] = 'Arial'

    # Create the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Set a dark background with backlighting effect
    fig.patch.set_facecolor('#000000')
    ax1.set_facecolor('#0f0f0f')
    ax2.set_facecolor('#0f0f0f')
    ax3.set_facecolor('#0f0f0f')

    neon_colors = ['#FF00FF', '#00FFFF', '#FFFF00']

    # Plot the stock price and moving averages with smooth curves and clean trend lines
    ax1.plot(stock_prices, label='Price', color=neon_colors[0], linewidth=2, linestyle='-', alpha=0.8)
    ax1.plot(sma50, label='50-Day SMA', color=neon_colors[1], linewidth=2, linestyle='--', alpha=0.8)
    ax1.plot(sma200, label='200-Day SMA', color=neon_colors[2], linewidth=2, linestyle='-.', alpha=0.8)
    ax1.set_title('Stock Price and Moving Averages', fontsize=16, color='#FFFFFF')
    ax1.legend()
    ax1.grid(True, color='#404040', linestyle='-', linewidth=0.8, alpha=0.5)

    # Plot the MACD line, signal line, and histogram with smooth curves and clean trend lines
    ax2.plot(macd, label='MACD', color=neon_colors[0], linewidth=2, alpha=0.8)
    ax2.plot(signal_line, label='Signal Line', color=neon_colors[1], linewidth=2, alpha=0.8)
    ax2.bar(stock_prices.index, macd - signal_line, label='MACD Histogram', color=neon_colors[2], alpha=0.5)
    ax2.set_title('MACD', fontsize=16, color='#FFFFFF')
    ax2.legend()
    ax2.grid(True, color='#404040', linestyle='-', linewidth=0.8, alpha=0.5)

    # Plot the RSI line and overbought/oversold lines with smooth curves and clean trend lines
    ax3.plot(rsi, label='RSI', color=neon_colors[0], linewidth=2, alpha=0.8)
    ax3.axhline(70, linestyle='--', color=neon_colors[1], label='Overbought', linewidth=2, alpha=0.8)
    ax3.axhline(30, linestyle='--', color=neon_colors[2], label='Oversold', linewidth=2, alpha=0.8)
    ax3.set_title('RSI', fontsize=16, color='#FFFFFF')
    ax3.legend()
    ax3.grid(True, color='#404040', linestyle='-', linewidth=0.8, alpha=0.5)

    # Add a subtle glow effect to the plot elements
    for ax in (ax1, ax2, ax3):
        for spine in ax.spines.values():
            spine.set_edgecolor(mcolors.CSS4_COLORS['black'])
            spine.set_linewidth(1.5)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(10)
            label.set_color('#FFFFFF')
        ax.tick_params(axis='both', colors='#FFFFFF')
        ax.xaxis.label.set_color('#FFFFFF')
        ax.yaxis.label.set_color('#FFFFFF')

    # Adjust layout spacing
    fig.tight_layout(pad=3)

    # Save the figure as a PNG image in memory
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', facecolor='#000000')
    buffer.seek(0)
    plt.close(fig)

    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def predict_volatility(ticker):
    try:
        stock_data, index_data = download_data(ticker)
        combined_data = clean_and_prepare_data(stock_data, index_data)
        volatility_data = calculate_volatility(combined_data)
        X = volatility_data['Volatility'].values[:-1].reshape(-1, 1)
        y = (volatility_data['Volatility'].shift(-1) > volatility_data['Volatility']).iloc[:-1].astype(int)

        # Use RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        future_volatility_change = model.predict([X[-1]])
        current_volatility = volatility_data['Volatility'].iloc[-1]
        sharpe_ratio = calculate_sharpe_ratio(combined_data['Stock_Returns'])
        sortino_ratio = calculate_sortino_ratio(combined_data['Stock_Returns'])
        beta = calculate_beta(combined_data['Stock_Returns'], combined_data['Index_Returns'])
        alpha = calculate_alpha(combined_data['Stock_Returns'], combined_data['Index_Returns'])

        # Updated part to use the latest close price
        latest_close_price = stock_data['Close'].iloc[-1]
        VaR = value_at_risk(latest_close_price, 0.95, current_volatility)
        ES = expected_shortfall(latest_close_price, 0.95, current_volatility)

        macd, signal_line = calculate_macd(stock_data['Close'])
        rsi = calculate_rsi(stock_data['Close'])
        bullish_signal = (macd[-1] > signal_line[-1]) and (rsi[-1] < 70 and rsi[-1] > 30)
        plot_url = plot_stock_data(stock_data['Close'])
        return {
            'current_volatility': current_volatility,
            'predicted_future_volatility_change': 'Increase' if future_volatility_change[0] else 'Decrease',
            'VaR': VaR,
            'ES': ES,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'beta': beta,
            'alpha': alpha,
            'market_sentiment': 'Bullish' if bullish_signal else 'Bearish',
            'plot_url': plot_url
        }
    except Exception as e:
        print(f"Error in predict_volatility function: {e}", file=sys.stderr)
        return None
