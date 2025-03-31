import pandas as pd
import numpy as np

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
        df[column] = df[column].replace({'9999-99-99' : 'NaN'})
        df[column] = df[column].replace({'?': 'NaN'})
    return df

def transform_data_type_bool(df):
    return df

def mapping_bool_values(df: pd.DataFrame, excluded_columns: list) -> pd.DataFrame:   
    for column in df.columns:
        if column not in excluded_columns:
            df[column] = df[column].astype(np.float64).astype(str)
            if column != 'SEX':
                df[column] = df[column].replace({'1.0': '1', '2.0': '0'})
            df[column] = df[column].astype(np.float64)
            
    # df['PREGNANT'] = df['PREGNANT'].fillna(0.0)
    
    return df

def remove_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns, axis=1)