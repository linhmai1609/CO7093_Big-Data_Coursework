import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfbefore = pd.read_csv('../data/Dataset.csv', low_memory=False)
df = pd.read_csv('../data/Dataset_revised.csv', low_memory=False)


rowsbefore, colsbefore = dfbefore.shape
print("Summary statistics before cleansing:", dfbefore.describe())
print(f"The dataset has {rowsbefore} rows and {colsbefore} columns.")


print("Summary statistics after cleansing:", df.describe())

rows, cols = df.shape
print(f"The dataset has {rows} rows and {cols} columns.")








def ICU_against_age():
    icu_counts = df.groupby('AGE')['ICU'].sum()
    total_counts = df.groupby('AGE')['ICU'].count()


    valid_ages = total_counts[total_counts > 0].index
    percent_in_icu = (icu_counts[valid_ages] / total_counts[valid_ages]) * 100


    plt.figure(figsize=(12, 6))
    plt.plot(percent_in_icu.index, percent_in_icu.values, marker='o')
    plt.xticks(np.arange(0, 115, step=5))
    plt.title('Percentage of Patients Admitted to ICU by Age')
    plt.xlabel('Age')
    plt.ylabel('Percentage in ICU (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

ICU_against_age()

def ICU_against_age_barchart():
   
    age_bins = pd.cut(df['AGE'], bins=range(0, df['AGE'].max() + 5, 5), right=False)
    df['AGE_GROUP'] = age_bins


    icu_grouped = df.groupby('AGE_GROUP', observed=True)['ICU'].sum()
    total_grouped = df.groupby('AGE_GROUP', observed=True)['ICU'].count()


    valid_groups = total_grouped[total_grouped > 0].index
    percent_by_group = (icu_grouped[valid_groups] / total_grouped[valid_groups]) * 100


    plt.figure(figsize=(14, 6))
    percent_by_group.plot(kind='bar')
    plt.title('Percentage of Patients Admitted to ICU by Age Group (5-Year Bins)')
    plt.xlabel('Age Group')
    plt.ylabel('Percentage in ICU (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

ICU_against_age_barchart()