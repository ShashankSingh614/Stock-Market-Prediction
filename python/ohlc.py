from modelmainaapl import getrnnmodelaapl
from modelmainmsft import getrnnmodelmsft
from modelmaingoog import getrnnmodelgoog
from modelmainamzn import getrnnmodelamzn
from modelmainnvda import getrnnmodelnvda
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def feature_of_company(company_name):
    FEATURES = ['Low', 'High', 'Open', 'Close', 'Volume']

    fig, axes = plt.subplots(nrows=len(FEATURES), ncols=1, figsize=(16, 8 * len(FEATURES)))

    for i, feature in enumerate(FEATURES):
        print("=" * 90)
        print(f"\nRunning model for feature: {feature}\n")
        
        if company_name == 'AAPL':
            y_pred, date, y_test, MAE, MAPE, MDAPE,r2,predicted_price, change_percent = getrnnmodelaapl(feature)
        elif company_name == 'MSFT':
            y_pred, date, y_test, MAE, MAPE, MDAPE,r2,predicted_price, change_percent= getrnnmodelmsft(feature)
        elif company_name == 'GOOG':
            y_pred, date, y_test, MAE, MAPE, MDAPE,r2,predicted_price, change_percent = getrnnmodelgoog(feature)
        elif company_name == 'AMZN':
            y_pred, date, y_test, MAE, MAPE, MDAPE,r2,predicted_price, change_percent = getrnnmodelamzn(feature)
        elif company_name == 'NVDA':
            y_pred, date, y_test, MAE, MAPE, MDAPE,r2,predicted_price, change_percent = getrnnmodelnvda(feature)

        min_length = min(len(date), len(y_pred), len(y_test))
        date = date[:min_length]
        y_pred = y_pred[:min_length]
        y_test = y_test[:min_length]
        
        print('--------------------------------')
        print('Date :',date)
        print('Lenghth Date :',len(date))
        print()
        print('y_pred :',y_pred)
        print('Lenghth y_pred :',len(y_pred))
        print()
        print('y_test :',y_test)
        print('Lenghth y_test :',len(y_test))
        print('--------------------------------')
            
        print(f"MDAPE for {feature}: {MDAPE:.2f}%")
        print(f"MAE for {feature}: {MAE:.2f}")
        print(f"MAPE for {feature}: {MAPE:.2f}%")

        axes[i].plot(date, y_test, label='Actual', color='#090364')
        axes[i].plot(date,y_pred, label='Predicted', color='#1960EF')
        axes[i].set_title(f"{feature}")
        axes[i].legend()
        axes[i].set_xlabel('Date')

        model_save_path_with_params = f"models/NASDAQ_rnn_model_{feature}.h5"
        print(f"Model saved at: {model_save_path_with_params}")

        plus = '+'
        minus = '-'
        print(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')
        text_color = 'green' if change_percent > 0 else 'red'
        # Add the predicted price and up/down text to the graph
        axes[i].text(0.95, 0.40, f'Predicted Price: {predicted_price}', 
                    transform=axes[i].transAxes, verticalalignment='top', 
                    horizontalalignment='right', color=text_color, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Add up/down text to the graph
        up_down_text = 'Up' if change_percent > 0 else 'Down'
        axes[i].text(0.95, 0.35, f'Direction: {up_down_text}', 
                    transform=axes[i].transAxes, verticalalignment='top', 
                    horizontalalignment='right', color=text_color, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Add accuracy text to the graph
        axes[i].text(0.95, 0.30, f'MAE: {MAE:.2f}', transform=axes[i].transAxes, 
             verticalalignment='top', horizontalalignment='right', 
             color=text_color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[i].text(0.95, 0.25, f'MAPE: {MAPE:.2f}%', transform=axes[i].transAxes, 
                    verticalalignment='top', horizontalalignment='right', 
                    color=text_color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[i].text(0.95, 0.20, f'MDAPE: {MDAPE:.2f}%', transform=axes[i].transAxes, 
                    verticalalignment='top', horizontalalignment='right', 
                    color=text_color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
    plt.tight_layout()
    fig.suptitle(f"{company_name}")
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)
    else:
        print("Layout engine is None. Unable to set layout parameters.")
    plt.tight_layout()
    axes[0].set_ylabel('Amount ($)')
    axes[1].set_ylabel('Amount ($)')
    axes[2].set_ylabel('Amount ($)')
    axes[3].set_ylabel('Amount ($)')
    fig.savefig(f"{company_name}.png", bbox_inches='tight')
    plt.show()

companies = {
    'AAPL': 'AAPL',
    'MSFT': 'MSFT',
    'GOOG': 'GOOG',
    'AMZN': 'AMZN',
    'NVDA': 'NVDA'
}

def get_company_data(selected_company):
    symbolcompany = companies[selected_company]
    feature_of_company(symbolcompany)