from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, NearMiss
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score

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
    
    model = DecisionTreeClassifier(max_depth=7, random_state=42)
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

def logistic_regression(training_set: dict , test_set: dict, solver: str, class_weight) -> None:
    # Initialize and fit model
    model = LogisticRegression(solver=solver, class_weight=class_weight, random_state=42)  # Increase max_iter if convergence issues
    model.fit(training_set['X'], training_set['Y'])

    # Predictions
    y_pred = model.predict(test_set['X'])
    y_prob = model.predict_proba(test_set['X'])[:, 1]  # Probabilities for ROC AUC

    # Evaluate
    print("Accuracy:", accuracy_score(test_set['Y'], y_pred))
    print("Classification Report:\n", classification_report(test_set['Y'], y_pred))

    # Coefficients
    coef_df = pd.DataFrame({'Variable': training_set['X'].columns, 'Coefficient': model.coef_[0]})
    print("Coefficients:\n", coef_df)

    # Cross-validation (using 5-fold by default)
    cv_scores = cross_val_score(model, training_set['X'], training_set['Y'], cv=5, scoring='accuracy')
    print("\nCross-Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy Score:", cv_scores.mean())
    print("Standard Deviation of CV Scores:", cv_scores.std())

    # Detailed cross-validation metrics
    scoring = ['precision', 'recall', 'f1', 'roc_auc']  # Metrics for class 1 (ICU admitted)
    cv_results = cross_validate(model, training_set['X'], training_set['Y'], cv=5, scoring=scoring)

    # Print mean and std for each metric
    print("\nCross-Validation Detailed Metrics (for class 1: ICU admitted):")
    print("Mean Precision:", cv_results['test_precision'].mean())
    print("Precision Std:", cv_results['test_precision'].std())
    print("Mean Recall:", cv_results['test_recall'].mean())
    print("Recall Std:", cv_results['test_recall'].std())
    print("Mean F1-Score:", cv_results['test_f1'].mean())
    print("F1-Score Std:", cv_results['test_f1'].std())
    print("Mean ROC AUC:", cv_results['test_roc_auc'].mean())
    print("ROC AUC Std:", cv_results['test_roc_auc'].std())

    # Overall test set metrics for class 1
    print("\nOverall Test Set Metrics (for class 1: ICU admitted):")
    print("Precision:", precision_score(test_set['Y'], y_pred))
    print("Recall:", recall_score(test_set['Y'], y_pred))
    print("F1-Score:", f1_score(test_set['Y'], y_pred))
    print("ROC AUC Score:", roc_auc_score(test_set['Y'], y_prob))