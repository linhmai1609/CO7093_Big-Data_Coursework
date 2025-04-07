import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, NearMiss
import pandas as pd
import findspark
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
findspark.init()
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from pyspark.ml.linalg import Vectors
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc




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

def trainAndTestModel(X_train, X_Test, y_train, y_test):
    model = LogisticRegression(max_iter=2000, random_state=42)

    model.fit(X_train, y_train)

    yPredictTrain = model.predict(X_train)
    yPredictTest = model.predict(X_Test)

    trainAccuracyScore = accuracy_score(y_train, yPredictTrain)
    testAccuracyScore = accuracy_score(y_test, yPredictTest)

    confusionMatrix = confusion_matrix(y_test, yPredictTest)
    report = classification_report(y_test, yPredictTest)

    crossValidationScores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    averageCrossValidationScore = crossValidationScores.mean()

    y_predict_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_predict_prob)

    fpr, sensitivity, _ = roc_curve(y_test, y_predict_prob, pos_label=1)
    auc = roc_auc_score(y_test, y_predict_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, sensitivity, label=f"ROC Curve (AUC = {auc:.3f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle='--', color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve for Logistic Regression")
    plt.legend(loc="lower right")
    plt.show()

    print(f"\nModel: Logistic Regression")
    print(f"Training Accuracy: {trainAccuracyScore}")
    print(f"Test Accuracy: {testAccuracyScore}")
    print(f"Cross-Validation Accuracy: {averageCrossValidationScore}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"\nConfusion Matrix:\n{confusionMatrix}")
    print(f"\nClassification Report:\n{report}")

    return model

def cleanDataForModeling(df: pd.DataFrame) -> pd.DataFrame:

    df = df.select_dtypes(include=[np.number])

    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Dataset_revised_new.csv')
df = pd.read_csv(DATA_PATH)

# One clean before running to ensure only numeric values
df = cleanDataForModeling(df)

#all feature selection
#selectedFeatures = feature_selection_mutual_info(df, target_column='ICU', select_k_best=10)
# selectedFeatures = feature_selection_coefficient(df, target_column='ICU', select_k_best=10)
#selectedFeatures = feature_selection_anova(df, target_column='ICU', select_k_best=10)
#selectedFeatures = feature_selection_chi2(df, target_column='ICU', continuous_columns=['AGE'], select_k_best=10)
selectedFeatures = feature_selection_rfe(df, target_column='ICU', select_k_best=10)

df_filtered = df[selectedFeatures + ['ICU']]

X_train, X_test, y_train, y_test = generate_train_test_over_set(df_filtered, 'ICU', test_size=0.2)

trainedModel = trainAndTestModel(X_train, X_test, y_train, y_test)

print("")
print("Moving to improved Model:")
print("")

#need these again here
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def trainAndTestImprovedModel():
    spark = SparkSession.builder.appName("Improved_Model").getOrCreate()

    spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    df.printSchema()
    df.summary().show()

    # Drop non-numeric columns
    df = df.drop("DATE_DIED")

    # filling missing values with mean
    for column, dtype in df.dtypes:
        if dtype in ["int", "double"]:
            mean_value = df.select(mean(col(column))).collect()[0][0]
            df = df.na.fill(mean_value, subset=[column])

    feature_columns = [col for col in df.columns if col != 'ICU']

    vectorAssembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    data_df  = vectorAssembler.transform(df).select('features', 'ICU')

    train_data, test_data = data_df.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(featuresCol='features', labelCol='ICU', maxIter=100)
    lrModel = lr.fit(train_data)
    predictions = lrModel.transform(test_data)

    y_true = predictions.select('ICU').collect()
    y_pred = predictions.select('prediction').collect()

    y_true = [int(row['ICU']) for row in y_true]
    y_pred = [int(row['prediction']) for row in y_pred]

    trainAccuracy = lrModel.summary.accuracy
    testAccuracy = accuracy_score(y_true, y_pred)

    evaluator = BinaryClassificationEvaluator(labelCol='ICU', metricName='areaUnderROC')
    roc_auc = evaluator.evaluate(predictions)

    print(f"Training Accuracy: {trainAccuracy}")
    print(f"Test Accuracy: {testAccuracy}")
    print(f"ROC AUC Score: {roc_auc}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    predictions_pd = predictions.select("probability", "ICU").toPandas()
    predictions_pd['probability'] = predictions_pd['probability'].apply(lambda x: float(x[1]))

    fpr, sensitivity, _ = metrics.roc_curve(predictions_pd['ICU'], predictions_pd['probability'])
    auc = metrics.auc(fpr, sensitivity)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, sensitivity, label=f"ROC Curve (AUC = {auc:.3f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle='--', color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve for improved Logistic Regression model")
    plt.legend(loc="lower right")
    plt.show()

    spark.stop()

trainAndTestImprovedModel()

print("")
print("Moving to K-Means clustering Model:")
print("")


def trainAndTestKmeansClassifiers():
    spark = SparkSession.builder.appName("KMeans").getOrCreate()
    spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

    df = df.drop("DATE_DIED")

    # Fill missing values with the mean of each column
    for column, dtype in df.dtypes:
        if dtype in ["int", "double"]:
            mean_value = df.select(mean(column)).collect()[0][0]
            df = df.na.fill(mean_value, subset=[column])

    feature_columns = [col for col in df.columns if col != 'ICU']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    df = assembler.transform(df)

    # Convert Spark DataFrame to Pandas DataFrame for visualisation purposes
    features_list = df.select("features").rdd.map(lambda row: row["features"].toArray()).collect()
    features_array = np.array(features_list)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_array)

    #Using PCA to reduce it for visualisation purposes
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    df_pandas = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', data=df_pandas, s=80, edgecolor="black", alpha=0.8)
    plt.title("Scatter Plot Before Clustering", fontsize=16, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.show()

    elbow_model = KElbowVisualizer(KMeans(n_init="auto", random_state=42), k=(2, 10), metric='distortion',
                                   timings=False)
    elbow_model.fit(reduced_features)
    elbow_model.show()

    optimal_k = 4

    kmeans = KMeans(n_clusters=optimal_k, n_init="auto", algorithm="lloyd", random_state=42)
    clusterLabels = kmeans.fit_predict(reduced_features)

    df_pandas['Cluster'] = clusterLabels

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pandas, palette="viridis", s=80, edgecolor="black",
                    alpha=0.8)
    plt.title("K-Means Clustering Result", fontsize=16, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(title="Clusters", fontsize=12, title_fontsize=13)
    plt.show()

    #building local classifiers
    originalDataFrame = pd.DataFrame(scaled_features)
    originalDataFrame['ICU'] = df.select('ICU').rdd.flatMap(lambda x: x).collect()
    originalDataFrame['Cluster'] = clusterLabels
    results = {}
    for cluster_num in sorted(originalDataFrame['Cluster'].unique()):
        print(f"Training and evaluating model for Cluster {cluster_num}...")

        clusterData = originalDataFrame[originalDataFrame['Cluster'] == cluster_num]
        clusterFeatures = clusterData.drop(columns=['ICU', 'Cluster'])
        clusterTarget = clusterData['ICU']

        X_train, X_test, y_train, y_test = train_test_split(clusterFeatures, clusterTarget, test_size=0.2,
                                                            random_state=42)

        model = SklearnLogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        #storing in results
        results[cluster_num] = {
            "Accuracy": accuracy,
            "AUC": auc,
            "Confusion Matrix": conf_matrix,
            "Classification Report": report
        }

        # Print results for this cluster
        print(f"\nCluster {cluster_num} - Accuracy: {accuracy}")
        print(f"Cluster {cluster_num} - AUC: {auc}")
        print(f"Cluster {cluster_num} - Confusion Matrix:\n{conf_matrix}")
        print(f"Cluster {cluster_num} - Classification Report:\n{report}")
    spark.stop()
    return results


trainAndTestKmeansClassifiers()
