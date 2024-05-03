# %%
from flask import Flask, render_template, request, redirect, url_for, flash
import model
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('stockselector.html')

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Received POST request for prediction.")
    ticker = request.form.get('ticker').upper().strip()

    if not ticker:
        flash('Please enter a valid stock ticker.', 'error')
        return redirect(url_for('index'))
    
    try:
        results = model.predict_volatility(ticker)
        if results is None:
            raise ValueError("No results returned. Please check the input ticker and data availability.")

        # Format results for display
        results['current_volatility'] = f"{results['current_volatility']:.2%}" if results['current_volatility'] is not None else "N/A"
        results['VaR'] = f"${results['VaR']:,.2f}" if results['VaR'] is not None else "N/A"
        results['ES'] = f"${results['ES']:,.2f}" if results['ES'] is not None else "N/A"
        results['sharpe_ratio'] = f"{results['sharpe_ratio']:.2f}" if results['sharpe_ratio'] is not None else "N/A"
        results['sortino_ratio'] = f"{results['sortino_ratio']:.2f}" if results['sortino_ratio'] is not None else "N/A"
        results['beta'] = f"{results['beta']:.2f}" if results['beta'] is not None else "N/A"
        results['alpha'] = f"{results['alpha']:.2f}" if results['alpha'] is not None else "N/A"
        results['market_sentiment'] = results['market_sentiment'] if 'market_sentiment' in results else "N/A"

        return render_template('results.html', results=results, ticker=ticker)
    except Exception as e:
        app.logger.error(f"Exception: {e}")
        flash(f'An error occurred: {e}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
