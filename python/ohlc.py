import matplotlib.pyplot as plt
import numpy as np
from modelmainaapl import getrnnmodelaapl
from modelmainmsft import getrnnmodelmsft
from modelmaingoog import getrnnmodelgoog
from modelmainamzn import getrnnmodelamzn
from modelmainnvda import getrnnmodelnvda

def feature_of_company(company_name):
    FEATURES = ['Low', 'HIGH', 'Open', 'Close', 'Volume']

    fig, axes = plt.subplots(nrows=len(FEATURES), ncols=1, figsize=(16, 8 * len(FEATURES)))

    for i, feature in enumerate(FEATURES):
        print("=" * 90)
        print(f"\nRunning model for feature: {feature}\n")
        
        if company_name=='AAPL':
            final_accuracy_scalar, y_pred, y_train, y_test, MAE, MAPE, MDAPE, df_union_zoom= getrnnmodelaapl(feature)
        elif company_name=='MSFT':
            final_accuracy_scalar, y_pred, y_train, y_test, MAE, MAPE, MDAPE, df_union_zoom= getrnnmodelmsft(feature)
        elif company_name=='GOOG':
            final_accuracy_scalar, y_pred, y_train, y_test, MAE, MAPE, MDAPE, df_union_zoom= getrnnmodelgoog(feature)
        elif company_name=='AMZN':
            final_accuracy_scalar, y_pred, y_train, y_test, MAE, MAPE, MDAPE, df_union_zoom= getrnnmodelamzn(feature)
        elif company_name=='NVDA':
            final_accuracy_scalar, y_pred, y_train, y_test, MAE, MAPE, MDAPE, df_union_zoom= getrnnmodelnvda(feature)

        min_size = min(len(df_union_zoom), len(y_pred), len(y_test))

        df_union_zoom = df_union_zoom.iloc[:min_size]
        y_pred = y_pred[:min_size]
        y_test = y_test[:min_size]
        
        print(f"Final accuracy for {feature}: {final_accuracy_scalar:.2f}%")
        print(f"MDAPE for {feature}: {MDAPE:.2f}%")
        print(f"MAE for {feature}: {MAE:.2f}")
        print(f"MAPE for {feature}: {MAPE:.2f}%")

        x_range = np.arange(len(y_test))
        axes[i].plot(x_range, y_test, label='Actual', color='#090364')
        axes[i].plot(x_range, y_pred, label='Predicted', color='#1960EF')
        axes[i].set_title(f"{feature}")
        axes[i].legend()
        axes[i].set_xlabel('Date')

        model_save_path_with_params = f"models/NASDAQ_rnn_model_{feature}.h5"
        print(f"Model saved at: {model_save_path_with_params}")

        accuracy_text = f"Final Acc: {final_accuracy_scalar:.2f}%"
        axes[i].text(0.95, 0.90, accuracy_text, transform=axes[i].transAxes,verticalalignment='top', horizontalalignment='right', color='green',bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        print("=" * 90)
        
    plt.tight_layout()
    fig.suptitle(f"{stockname}") 
    fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)
    plt.tight_layout()
    axes[0].set_ylabel('Amount ($)')
    axes[1].set_ylabel('Amount ($)')
    axes[2].set_ylabel('Amount ($)')
    axes[3].set_ylabel('Amount ($)')
    fig.savefig(f"{stockname}.png") 
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