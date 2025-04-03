from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, NearMiss
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

def generate_train_test_over_set(df: pd.DataFrame, target_column: str, test_size: float) -> tuple:
    y = df[target_column]
    X = df.drop(target_column, axis=1)

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

    # Applying NearMiss to the training data
    nm = NearMiss(version=1)  # Can use version=1, 2, or 3
    X_train, y_train = nm.fit_resample(X_train, y_train)
    print(f'X_train shape after applying Tomek Links: {X_train.shape}')
    print(f'y_train shape after applying Tomek Links: {y_train.shape}')

    return X_train, X_test, y_train, y_test

def feature_selection_coefficient(df: pd.DataFrame, target_column: str, select_k_best: int) -> None:
    # Discretize features into bins (e.g., 5 bins)
    y = df[target_column]
    X = df.drop(target_column, axis=1)

    # Calculate correlation with each feature
    correlations = X.corrwith(pd.Series(y))
    print("Correlations with target:")
    print(correlations.sort_values(ascending=False))

    # Select top k features (e.g., top 2)
    top_features = correlations.abs().sort_values(ascending=False).head(select_k_best).index
    print(f"Top {select_k_best} features:", top_features.tolist())
    return top_features.tolist()

def feature_selection_mutual_info(df: pd.DataFrame, target_column: str, select_k_best: int) -> None:
    y = df[target_column]
    X = df.drop(target_column, axis=1)

    # Apply Mutual Information
    mi_selector = SelectKBest(mutual_info_classif, k=select_k_best)  # Select top 5 features
    X_new = mi_selector.fit_transform(X, y)

    # Get scores and selected features
    mi_scores = pd.Series(mi_selector.scores_, index=X.columns)
    print("Mutual Information Scores:")
    print(mi_scores.sort_values(ascending=False))

    selected_features = X.columns[mi_selector.get_support()].tolist()
    print("Selected features:", selected_features)
    return selected_features

def feature_selection_coefficient(df: pd.DataFrame, target_column: str, select_k_best: int) -> None:
    y = df[target_column]
    X = df.drop(target_column, axis=1)

    # Calculate correlation with each feature
    correlations = X.corrwith(pd.Series(y))
    print("Correlations with target:")
    print(correlations.sort_values(ascending=False))

    # Select top k features (e.g., top 2)
    top_features = correlations.abs().sort_values(ascending=False).head(select_k_best).index
    print(f"Top {select_k_best} features:", top_features.tolist())
    return top_features.tolist()

def feature_selection_anova(df: pd.DataFrame, target_column: str, select_k_best: int) -> None:
    y = df[target_column]
    X = df.drop(target_column, axis=1)

    # Apply ANOVA F-Test
    anova_selector = SelectKBest(f_classif, k=select_k_best)  # Select top 2 features
    X_new = anova_selector.fit_transform(X, y)

    # Get scores and selected features
    anova_scores = pd.Series(anova_selector.scores_, index=X.columns)
    print("ANOVA F-Scores:")
    print(anova_scores.sort_values(ascending=False))

    selected_features = X.columns[anova_selector.get_support()].tolist()
    print("Selected features:", selected_features)
    return selected_features

def feature_selection_chi2(df: pd.DataFrame, target_column: str, continuous_columns: list, select_k_best: int) -> None:
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    for col in continuous_columns:
        X[col] = pd.cut(X[col], bins=10, labels=False)

    # Apply Chi-Square test
    chi2_selector = SelectKBest(chi2, k=select_k_best)  # Select top 2 features
    X_new = chi2_selector.fit_transform(X, y)

    # Get scores and selected features
    chi2_scores = pd.Series(chi2_selector.scores_, index=X.columns)
    print("Chi-Square Scores:")
    print(chi2_scores.sort_values(ascending=False))

    selected_features = X.columns[chi2_selector.get_support()].tolist()
    print("Selected features:", selected_features)
    return selected_features

def feature_selection_rfe(df: pd.DataFrame, target_column: str, select_k_best: int) -> None:
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    # RFE using sklearn
    rfe = RFE(estimator=model, n_features_to_select=select_k_best)  # Select 2 features

    # Fit RFE
    rfe = rfe.fit(X, y)

    # Results
    print("Recursive Feature Elimination:")
    selected_features = X.columns[rfe.support_].tolist()
    print("Selected features:", selected_features)
    print("Feature ranking (1 = selected):", rfe.ranking_)

    # Evaluate performance
    X_rfe = X[selected_features]
    score = cross_val_score(model, X_rfe, y, cv=select_k_best, scoring='accuracy').mean()
    print("Accuracy with selected features:", score)

    return selected_features