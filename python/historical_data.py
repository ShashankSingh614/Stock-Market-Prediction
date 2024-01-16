import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import os

def get_stock_data(soup, data_test_value):
    element = soup.find('td', {'data-test': data_test_value})
    if element is not None:
        return element.text.strip()
    else:
        return "N/A"

def get_high_low_prices(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'Time Series (Daily)' in data:
        daily_data = data['Time Series (Daily)']
        latest_date = list(daily_data.keys())[0]
        high_price = daily_data[latest_date]['2. high']
        low_price = daily_data[latest_date]['3. low']
        return high_price, low_price
    else:
        return "N/A", "N/A"

companies = {
    'AAPL': 'AAPL',
    'MSFT': 'MSFT',
    'GOOG': 'GOOG',
    'AMZN': 'AMZN',
    'NVDA': 'NVDA'
}

base_url = "https://finance.yahoo.com/quote/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

data_folder = 'historical_data'
os.makedirs(data_folder, exist_ok=True)

def get_historical_data(selected_company):
    symbol = companies[selected_company]

    url = f"{base_url}{symbol}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    previous_close = get_stock_data(soup, 'PREV_CLOSE-value')
    open_price = get_stock_data(soup, 'OPEN-value')
    bid = get_stock_data(soup, 'BID-value')
    ask = get_stock_data(soup, 'ASK-value')
    days_range = get_stock_data(soup, 'DAYS_RANGE-value')
    week_52_range = get_stock_data(soup, 'FIFTY_TWO_WK_RANGE-value')
    volume = get_stock_data(soup, 'TD_VOLUME-value')
    avg_volume = get_stock_data(soup, 'AVERAGE_VOLUME_3MONTH-value')
    market_cap = get_stock_data(soup, 'MARKET_CAP-value')
    beta = get_stock_data(soup, 'BETA_5Y-value')
    pe_ratio = get_stock_data(soup, 'PE_RATIO-value')
    eps_ratio = get_stock_data(soup, 'EPS_RATIO-value')
    earnings_date = get_stock_data(soup, 'EARNINGS_DATE-value')
    dividend_yield = get_stock_data(soup, 'DIVIDEND_AND_YIELD-value')
    ex_dividend_date = get_stock_data(soup, 'EX_DIVIDEND_DATE-value')
    target_est = get_stock_data(soup, 'ONE_YEAR_TARGET_PRICE-value')

    alpha_vantage_api_key = "YOUR_ALPHA_VANTAGE_API_KEY"
    high_price, low_price = get_high_low_prices(symbol, alpha_vantage_api_key)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        'Time': [current_time],
        'Previous Close': [previous_close],
        'Open Price': [open_price],
        'Bid': [bid],
        'Ask': [ask],
        'Day\'s Range': [days_range],
        '52-Week Range': [week_52_range],
        'Volume': [volume],
        'Avg. Volume': [avg_volume],
        'Market Cap': [market_cap],
        'Beta (5Y Monthly)': [beta],
        'PE Ratio (TTM)': [pe_ratio],
        'EPS (TTM)': [eps_ratio],
        'Earnings Date': [earnings_date],
        'Forward Dividend & Yield': [dividend_yield],
        'Ex-Dividend Date': [ex_dividend_date],
        '1y Target Est': [target_est],
        'High Price': [high_price],
        'Low Price': [low_price]
    }

    stock_data_df = pd.DataFrame(data)

    file_path = os.path.join('historical_data', f"{symbol}_historical_data.csv")

    stock_data_df.to_csv(file_path, index=False)

    print(f"Historical data for {symbol} saved successfully.")