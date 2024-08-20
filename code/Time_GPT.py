import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
from dotenv import load_dotenv
from nixtla import NixtlaClient


load_dotenv()
nixtla_client = NixtlaClient(
    api_key = os.environ.get("NIXTLA_API_KEY")
)


def get_config(key):
    with open("/Users/bhargavi/PycharmProjects/LLMs_for_Macroeconomics_Variable/config.json", 'r') as f:
        config = json.load(f)
        value = config[key]
        return value


df = pd.read_csv(get_config("train_dataset"))
df.rename(columns={"DATE": "ds"}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])
df.rename(columns={"IRSTPI01INM156N": "y"}, inplace=True)

print(df.info())

fcst_df = nixtla_client.forecast(df, h=12, level=[80, 90])

# nixtla_client.plot(df, fcst_df, level=[80, 90])

print(fcst_df)




#####################
#evaluate
#####################
test_df = pd.read_csv(get_config("test_dataset"))
actual = test_df['IRSTPI01INM156N'].values
predicted = fcst_df['TimeGPT'].values
# Mean Absolute Error
mae = np.mean(np.abs(actual - predicted))

# Mean Squared Error
mse = np.mean((actual - predicted) ** 2)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Percentage Error
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Symmetric Mean Absolute Percentage Error
smape = np.mean(2.0 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
print(f"sMAPE: {smape}%")
