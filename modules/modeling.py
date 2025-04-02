from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_train_test_over_set(df: pd.DataFrame, target_column: str, test_size: float) -> tuple:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        , test_size=test_size
                                                        , random_state=42
                                                        , stratify=y)
    
    # Print the shape of the training and test data splits
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Applying SMOTE to the training data
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f'X_train shape after applying SMOTE: {X_train.shape}')
    print(f'y_train shape after applying SMOTE: {y_train.shape}')

    return X_train, X_test, y_train, y_test

def generate_train_test_under_set(df: pd.DataFrame, target_column: str, test_size: float) -> tuple:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        , test_size=test_size
                                                        , random_state=42
                                                        , stratify=y)
    
    # Print the shape of the training and test data splits
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Applying Tomek's links to the training data
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X_train, y_train)
    print(f'X_train shape after applying Tomek Links: {X_res.shape}')
    print(f'y_train shape after applying Tomek Links: {y_res.shape}')

    return X_train, X_test, y_train, y_test