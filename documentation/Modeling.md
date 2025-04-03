# Modeling process

## I/ Prepare dataset for training and testing: 
### 1. Speculation
- After finishing cleansing dataset:
    - There are 176,037 records which indicate ICU as False (90.581% of the total number of records)
    - There are 18,306 records which indicate ICU as True (9.419% of the total number of records)

&rarr;The dataset is imbalance (The ICU admission cases are outnumbered by the ICU non-admission ones)

### 2. Application
The following methods might be applied when generating training set and test set:
- Stratification will be applied make sure each subset (e.g., train/test split) reflects the overall distribution of the original data.
- Approach 1: Oversampling with SMOTE
    - Oversampling with SMOTE will be applied for the training dataset to ensures maximum data utilisation for the upcoming model.
- Approach 2: Undersampling with NearMiss due to these reasons:
    - Dataset has been cleansed previously, therefore methods that handle noise like Cluster Centroids and Edited Nearest Neighbors are not needed.
    - The author wants to limit the amount of data loss as much as possible, therefore methods that leads to large amount of data loss like Random Undersampling and Cluster Centroids
    - NearMiss has a good class separability 

## II/ 