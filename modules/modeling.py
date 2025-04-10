from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, NearMiss
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, learning_curve
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

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
    print(f'X_train shape after applying NearMiss {X_train.shape}')
    print(f'y_train shape after applying NearMiss: {y_train.shape}')

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

def cleanDataForModeling(df: pd.DataFrame) -> pd.DataFrame:

    df = df.select_dtypes(include=[np.number])

    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df


def logistic_regression_model(training_set: dict , test_set: dict, solver: str, class_weight):
    model = LogisticRegression(solver=solver, class_weight=class_weight, max_iter=2000, random_state=42)

    model.fit(training_set['X'], training_set['Y'])

    yPredictTrain = model.predict(training_set['X'])
    yPredictTest = model.predict(test_set['X'])

    trainAccuracyScore = accuracy_score(training_set['Y'], yPredictTrain)
    testAccuracyScore = accuracy_score(test_set['Y'], yPredictTest)

    confusionMatrix = confusion_matrix(test_set['Y'], yPredictTest)
    report = classification_report(test_set['Y'], yPredictTest)

    crossValidationScores = cross_val_score(model, training_set['X'], training_set['Y'], cv=5, scoring='accuracy')
    averageCrossValidationScore = crossValidationScores.mean()

    y_predict_prob = model.predict_proba(test_set['X'])[:, 1]
    roc_auc = roc_auc_score(test_set['Y'], y_predict_prob)

    fpr, sensitivity, _ = roc_curve(test_set['Y'], y_predict_prob, pos_label=1)
    auc = roc_auc_score(test_set['Y'], y_predict_prob)


    print(f"\nModel: Logistic Regression")
    print(f"Training Accuracy: {trainAccuracyScore}")
    print(f"Test Accuracy: {testAccuracyScore}")

    # Cross-validation (using 5-fold by default)
    print("\nCross-Validation Accuracy Scores:", crossValidationScores)
    print("Mean CV Accuracy Score:", averageCrossValidationScore)
    print("Standard Deviation of CV Scores:", crossValidationScores.std())

    print(f"ROC AUC Score: {roc_auc}")    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, sensitivity, label=f"ROC Curve (AUC = {auc:.3f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle='--', color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve for Logistic Regression")
    plt.legend(loc="lower right")
    plt.show()

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print(f"\nClassification Report:\n{report}")

    # Overall test set metrics for class 1
    print("\nOverall Test Set Metrics (for class 1: ICU admitted):")
    print("Precision:", precision_score(test_set['Y'], yPredictTest))
    print("Recall:", recall_score(test_set['Y'], yPredictTest))
    print("F1-Score:", f1_score(test_set['Y'], yPredictTest))
    print("ROC AUC Score:", roc_auc_score(test_set['Y'], y_predict_prob))

    return model

def random_forest_model(training_set: dict , test_set: dict, n_estimators: int, min_samples_split: int):
    # Create and train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators  # number of trees
                                        , min_samples_split= min_samples_split,
                                        #   max_depth=10,      # maximum depth of trees
                                        random_state=42)   # for reproducibility

    # Fit the model
    rf_classifier.fit(training_set['X'], training_set['Y'])

    # Make predictions
    y_pred = rf_classifier.predict(test_set['X'])

    # Calculate accuracy
    accuracy = accuracy_score(test_set['Y'], y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(test_set['Y'], y_pred))

    # Check for overfitting
    train_pred = rf_classifier.predict(training_set['X'])
    print(f"Training accuracy: {accuracy_score(training_set['Y'], train_pred):.4f}")
    print(f"Test accuracy: {accuracy_score(test_set['Y'], y_pred):.4f}")

    # Calculate accuracy
    accuracy = accuracy_score(test_set['Y'], y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Feature importance
    feature_importance = rf_classifier.feature_importances_
    print("\nFeature Importance:")
    for i, importance in enumerate(feature_importance):
        print(f"Feature {i+1}: {importance:.4f}")

    # Plot feature importances
    import matplotlib.pyplot as plt
    plt.bar(training_set['X'].columns, feature_importance)
    plt.xticks(rotation=45)
    plt.title("Feature Importances")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(test_set['Y'], y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Learning curves to assess if more data would help
    train_sizes, train_scores, test_scores = learning_curve(
        rf_classifier, training_set['X'], training_set['Y'], cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))

    # Plot learning curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Cross-validation score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()