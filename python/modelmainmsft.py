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
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})

end_date = date.today().strftime("%Y-%m-%d")
start_date = '2019-01-01'

stockname = 'Microsoft Corp'
symbol = 'MSFT'

import yfinance as yf
df = yf.download(symbol, start=start_date, end=end_date)

df_plot = df.copy()

ncols = 2
nrows = int(round(df_plot.shape[1] / ncols, 0))

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
for i, ax in enumerate(fig.axes):
        sns.lineplot(data = df_plot.iloc[:, i], ax=ax)
        ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.tight_layout()

# Add zoom-in features
def onselect(xmin, xmax):
    ax.set_xlim(xmin, xmax)
    fig.canvas.draw()

span = SpanSelector(ax, onselect, 'horizontal', useblit=True)


FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume']

data = pd.DataFrame(df)
data_filtered = data[FEATURES]

data_filtered_ext = data_filtered.copy()
def getrnnmodelmsft(xyz):
    print('Trainning Model for : ',xyz)
    print()
    data_filtered_ext['Prediction'] = data_filtered_ext[xyz]

    nrows = data_filtered.shape[0]

    np_data_unscaled = np.array(data_filtered)
    np_data = np.reshape(np_data_unscaled, (nrows, -1))

    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)

    scaler_pred = MinMaxScaler()
    df_Close = pd.DataFrame(data_filtered_ext[xyz])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)

    sequence_length = 50
    index_Close = data.columns.get_loc("Close")

    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)
    train_data = np_data_scaled[0:train_data_len, :]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]

    def partition_dataset(sequence_length, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:])
            y.append(data[i, index_Close])
        
        x = np.array(x)
        y = np.array(y)
        return x, y

    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)

    model_save_path_with_params = f"models/{stockname}_rnn_model_{xyz}.h5"

    if os.path.exists(model_save_path_with_params):
        model = tf.keras.models.load_model(model_save_path_with_params)
        print("Existing model loaded.")
    else:
        model = Sequential()
        n_neurons = x_train.shape[1] * x_train.shape[2]
        model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(n_neurons, return_sequences=False))
        model.add(Dense(5))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        epochs = 50
        batch_size = 16
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=[early_stop]
                        )
        print("New model trained for : ",xyz)
        
        model.save(model_save_path_with_params)
        print(f"New model saved at: {model_save_path_with_params}")

    y_pred_scaled = model.predict(x_test)
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

    MAE = mean_absolute_error(y_test_unscaled, y_pred)
    MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
    MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100

    display_start_date = "2023-11-01" 

    train = pd.DataFrame(data_filtered_ext[xyz][:train_data_len + 1]).rename(columns={xyz: 'y_train'})
    valid = pd.DataFrame(data_filtered_ext[xyz][train_data_len:]).rename(columns={xyz: 'y_test'})
    valid.insert(1, "y_pred", y_pred, True)
    valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
    df_union = pd.concat([train, valid])

    df_union_zoom = df_union[df_union.index > display_start_date]

    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title("y_pred vs y_test")
    plt.ylabel(stockname, fontsize=18)
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)

    df_sub = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom["residuals"].dropna()]
    ax1.bar(height=df_union_zoom['residuals'].dropna(), x=df_union_zoom['residuals'].dropna().index, width=3, label='residuals', color=df_sub)
    plt.legend()

    df_temp = df[-sequence_length:]
    new_df = df_temp.filter(FEATURES)

    N = sequence_length
    last_N_days = new_df[-sequence_length:].values
    last_N_days_scaled = scaler.transform(last_N_days)

    X_test_new = []
    X_test_new.append(last_N_days_scaled)

    pred_price_scaled = model.predict(np.array(X_test_new))
    pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

    price_today = np.round(new_df[xyz][-1], 2)
    predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
    change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)

    print("Predicted Price for ",xyz,"is : ",predicted_price)
    model.save(model_save_path_with_params)

    # Predict the closing price for the latest day
    predicted_final_price_scaled = model.predict(np.array([last_N_days_scaled]))
    predicted_final_price_unscaled = scaler_pred.inverse_transform(predicted_final_price_scaled.reshape(-1, 1))

    # Actual closing price of the latest day
    actual_final_price = new_df[xyz].iloc[-1]

    # Calculate the accuracy of the final prediction
    final_accuracy = 100 - np.abs((actual_final_price - predicted_final_price_unscaled) / actual_final_price) * 100

    # Extract the scalar value from the NumPy array
    final_accuracy_scalar = final_accuracy.item()

    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title("y_pred vs y_test")
    plt.ylabel(stockname, fontsize=18)

    # Plot predicted and actual values
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)

    # Add bars for residuals
    df_sub = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom["residuals"].dropna()]
    ax1.bar(height=df_union_zoom['residuals'].dropna(), x=df_union_zoom['residuals'].dropna().index, width=3, label='residuals', color=df_sub)

    # Add legend
    plt.legend(loc="upper left")

    # Add a secondary y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", fontsize=18)

    ax2.text(0.05, 0.9, f"MAE: {MAE:.2f}", transform=ax2.transAxes, color='red')
    ax2.text(0.05, 0.85, f"MAPE: {MAPE:.2f}%", transform=ax2.transAxes, color='red')
    ax2.text(0.05, 0.8, f"MDAPE: {MDAPE:.2f}%", transform=ax2.transAxes, color='red')

    # Print the final accuracy
    print(f"Final Accuracy: {final_accuracy_scalar:.2f}%")

    print(f"Model saved at: {model_save_path_with_params}")

    return final_accuracy_scalar, y_pred, scaler_pred.inverse_transform(y_train.reshape(-1, 1)), y_test_unscaled, MAE, MAPE, MDAPE, df_union_zoom