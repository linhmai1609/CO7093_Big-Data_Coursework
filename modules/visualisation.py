import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

dfbefore = pd.read_csv('../data/Dataset.csv', low_memory=False)
df = pd.read_csv('../data/Dataset_revised_new.csv', low_memory=False)


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

#ICU_against_age()

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

def plot_scatter_matrix():
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns for scatter matrix
    scatter_matrix(numeric_df, alpha=0.8, figsize=(10, 10), diagonal='hist')
    plt.show()

#plot_scatter_matrix()

def unit_against_ICU():
    df_filtered = df.dropna(subset=['MEDICAL_UNIT', 'ICU'])


    icu_counts = df_filtered.groupby(['MEDICAL_UNIT', 'ICU']).size().unstack(fill_value=0)

    icu_percentages = icu_counts.div(icu_counts.sum(axis=1), axis=0) * 100

    icu_percentages.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
    plt.title('ICU Admission Percentages by Medical Unit')
    plt.ylabel('Percentage')
    plt.xlabel('Medical Unit')
    plt.legend(title='ICU')
    plt.tight_layout()
    plt.show()

#unit_against_ICU()

def asthma_against_ICU():
    df_filtered = df.dropna(subset=['ICU', 'ASTHMA'])


    asthma_icu_counts = df_filtered.groupby(['ASTHMA', 'ICU']).size().unstack(fill_value=0)

    asthma_icu_percentages = asthma_icu_counts.div(asthma_icu_counts.sum(axis=1), axis=0) * 100

    asthma_icu_percentages.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='Set2')
    plt.title('ICU Admission %: With vs Without Asthma')
    plt.xlabel('Has Asthma')
    plt.ylabel('Percentage')
    plt.legend(title='ICU')
    plt.xticks(ticks=[0, 1], labels=['No Asthma', 'Has Asthma'], rotation=0)
    plt.tight_layout()
    plt.show()

#asthma_against_ICU()

def pneumonia_against_ICU():
    df_filtered = df.dropna(subset=['ICU', 'PNEUMONIA'])

    pneumonia_icu_counts = df_filtered.groupby(['PNEUMONIA', 'ICU']).size().unstack(fill_value=0)
    pneumonia_icu_percentages = pneumonia_icu_counts.div(pneumonia_icu_counts.sum(axis=1), axis=0) * 100

    pneumonia_icu_percentages.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='Set2')
    plt.title('ICU Admission %: With vs Without Pneumonia')
    plt.xlabel('Pneumonia vs No Pneumonia')
    plt.ylabel('Percentage')
    plt.legend(title='ICU')
    plt.xticks(ticks=[0, 1], labels=['No Pneumonia', 'Has Pneumonia'], rotation=0)
    plt.tight_layout()
    plt.show()
#pneumonia_against_ICU()

def pregnancy_against_ICU():
    df_filtered = df.dropna(subset=['ICU', 'PREGNANT'])

    preg_icu_counts = df_filtered.groupby(['PREGNANT', 'ICU']).size().unstack(fill_value=0)
    preg_icu_percentages = preg_icu_counts.div(preg_icu_counts.sum(axis=1), axis=0) * 100

    preg_icu_percentages.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='Set2')
    plt.title('ICU Admission %: With vs Without Pregnancy')
    plt.xlabel('Pregnant vs no Pregnancy')
    plt.ylabel('Percentage')
    plt.legend(title='ICU')
    plt.xticks(ticks=[0, 1], labels=['No Pregnancy', 'Is pregnant'], rotation=0)
    plt.tight_layout()
    plt.show()
pregnancy_against_ICU()

def diabetes_against_ICU():
    df_filtered = df.dropna(subset=['ICU', 'DIABETES'])

    diabetes_icu_counts = df_filtered.groupby(['PREGNANT', 'ICU']).size().unstack(fill_value=0)
    diabetes_icu_percentages = diabetes_icu_counts.div(diabetes_icu_counts.sum(axis=1), axis=0) * 100

    diabetes_icu_percentages.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='Set2')
    plt.title('ICU Admission %: With vs Without Diabetes')
    plt.xlabel('Diabetes vs no Diabetes')
    plt.ylabel('Percentage')
    plt.legend(title='ICU')
    plt.xticks(ticks=[0, 1], labels=['No Diabetes', 'Has diabetes'], rotation=0)
    plt.tight_layout()
    plt.show()

diabetes_against_ICU()


def plot_condition_icu_rate_scaled(col_name, labels, title):
    df_filtered = df.dropna(subset=[col_name, 'ICU'])


    condition_counts = df_filtered.groupby(col_name)['ICU'].value_counts(normalize=True).unstack().fillna(0) * 100

    if 1.0 in condition_counts.columns:
        icu_only = condition_counts[1.0]
    else:
        icu_only = pd.Series([0, 0], index=condition_counts.index)

    
    plt.figure(figsize=(6, 5))
    plt.bar([0, 1], icu_only.values, color=['#66c2a5', '#fc8d62'])
    plt.title(title)
    plt.ylabel('Chance of ICU Admission (%)')
    plt.xticks([0, 1], labels)
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.show()


plot_condition_icu_rate_scaled('ASTHMA', ['No Asthma', 'Has Asthma'], 'Chance of ICU Admission: Asthma vs No Asthma')
plot_condition_icu_rate_scaled('PNEUMONIA', ['No Pneumonia', 'Has Pneumonia'], 'Chance of ICU Admission: Pneumonia vs No Pneumonia')
plot_condition_icu_rate_scaled('PREGNANT', ['Not Pregnant', 'Pregnant'], 'Chance of ICU Admission: Pregnancy vs No Pregnancy')
plot_condition_icu_rate_scaled('DIABETES', ['No Diabetes', 'Has Diabetes'], 'Chance of ICU Admission: Diabetes vs No Diabetes')

