import json
import statsmodels.api as sm
import pandas as pd

def get_config(key):
    with open("/Users/bhargavi/PycharmProjects/LLMs_for_Macroeconomics_Variable/config.json", 'r') as f:
        config = json.load(f)
        value = config[key]
        return value

def train_test_split_time_series(data, test_horizon=12):
    """
    Splits the time series data into train and test sets.

    Parameters:
    - data: pandas DataFrame with a DateTime index and a single column for the interest rates.
    - test_horizon: int, the number of months to include in the test set.

    Returns:
    - train: pandas DataFrame, the training set.
    - test: pandas DataFrame, the test set.
    """
    # Calculate the index to split the data
    split_index = len(data) - test_horizon

    # Split the data into train and test sets
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    return train, test


def process_time_series(dataframe, column_name):
    """
    Process the time series data to compute L1 lag, apply HP trend, and CF cycle filters.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to process.

    Returns:
    pd.DataFrame: The updated DataFrame with additional columns.
    """
    processed_df = dataframe.copy()

    # Creating the lagged series (L1)
    processed_df[f'{column_name}_L1'] = processed_df[column_name].shift(1)

    # Dropping the first row due to NaN after lagging
    processed_df.dropna(inplace=True)

    # Applying HP filter to the L1 lagged series
    hp_cycle, hp_trend = sm.tsa.filters.hpfilter(processed_df[f'{column_name}_L1'])
    processed_df[f'{column_name}_hp_trend_L1'] = hp_trend.round(2).copy()

    # Applying CF filter to the L1 lagged series
    cf_cycle, cf_trend = sm.tsa.filters.cffilter(processed_df[f'{column_name}_L1'])
    processed_df[f'{column_name}_cf_cycle_L1'] = cf_cycle.round(2).copy()

    processed_df = processed_df[['DATE',column_name, f'{column_name}_L1',
                                 f'{column_name}_hp_trend_L1',
                                 f'{column_name}_cf_cycle_L1']]

    return processed_df


df = pd.read_csv(get_config("data_file_India"))
df['DATE'] = pd.to_datetime(df['DATE'])
print(df.info())

train_initial, test = train_test_split_time_series(df, test_horizon=get_config("horizon"))

#------------------- CP & HP ----------------

train = process_time_series(train_initial, get_config('column_name'))

print(train.head())

#---------------------- Save the data--------------
train.to_csv(f"{get_config('train_output_path')}/{get_config('country')}_{get_config('horizon')}_train.csv", index=False)
test.to_csv(f"{get_config('train_output_path')}/{get_config('country')}_{get_config('horizon')}_test.csv", index=False)

print(f"Train set saved to {get_config('train_output_path')}")
print(f"Test set saved to {get_config('test_output_path')}")

print("Train set:\n", train.tail())
print("Test set:\n", test)


