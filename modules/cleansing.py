import pandas as pd

def remove_outliers(df: pd.DataFrame, column) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
    return df

def remove_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        df[column] = df[column].replace({'?': 'NaN'}, {'9999-99-99' : 'NaN'})
    return df

def transform_data_type_bool(df):
    return df

def mapping_bool_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:   
    
    return df
