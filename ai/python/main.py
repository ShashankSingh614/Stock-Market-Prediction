import os
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from historical_data import get_historical_data, companies

companies = {
    1: 'AAPL',
    2: 'MSFT',
    3: 'GOOG',
    4: 'AMZN',
    5: 'NVDA'
}

def save_news_to_file(company_name, news_content):
    file_path = os.path.join('news', f'{company_name}.txt')
    with open(file_path, 'w') as file:
        file.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write(news_content)

def scrape_stock_news(company_symbol):
    url = f'https://finance.yahoo.com/quote/{company_symbol}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    news_articles = soup.find_all('li', class_='js-stream-content')
    news_content = '\n'.join([article.get_text() for article in news_articles])

    return news_content
while True:
    num = int(input("Select a Company (1-5): "))
    if num ==1:
        selected_company = companies[num]
        get_historical_data(selected_company)
        news_content = scrape_stock_news(selected_company)
        save_news_to_file(selected_company, news_content)
        time.sleep(30)
    elif num ==2:
        selected_company = companies[num]
        get_historical_data(selected_company)
        news_content = scrape_stock_news(selected_company)
        save_news_to_file(selected_company, news_content)
        time.sleep(30)
    elif num ==3:
        selected_company = companies[num]
        get_historical_data(selected_company)
        news_content = scrape_stock_news(selected_company)
        save_news_to_file(selected_company, news_content)
        time.sleep(30)
    elif num ==4:
        selected_company = companies[num]
        get_historical_data(selected_company)
        news_content = scrape_stock_news(selected_company)
        save_news_to_file(selected_company, news_content)
        time.sleep(30)
    elif num ==5:
        selected_company = companies[num]
        get_historical_data(selected_company)
        news_content = scrape_stock_news(selected_company)
        save_news_to_file(selected_company, news_content)
        time.sleep(30)
    else:
        print("Invalid selection. Please choose a number between 1 and 5.")