import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns


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


def plot_condition_icu_rate_scaled(col_name, labels, title):
    df_filtered = df.dropna(subset=[col_name, 'ICU'])

    condition_counts = df_filtered.groupby(col_name)['ICU'].value_counts(normalize=True).unstack().fillna(0) * 100

    if 1.0 in condition_counts.columns:
        icu_only = condition_counts[1.0]
    else:
        icu_only = pd.Series([0, 0], index=[0, 1])

    plot_df = pd.DataFrame({
        'Condition': labels,
        'ICU Rate (%)': icu_only.values
    })

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x='Condition', y='ICU Rate (%)', hue='Condition', data=plot_df,
                     palette=['#66c2a5', '#fc8d62'], legend=False)


    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=10)

    plt.title(title)
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.show()


plot_condition_icu_rate_scaled('ASTHMA', ['No Asthma', 'Has Asthma'], 'Chance of ICU Admission: Asthma vs No Asthma')
plot_condition_icu_rate_scaled('PNEUMONIA', ['No Pneumonia', 'Has Pneumonia'], 'Chance of ICU Admission: Pneumonia vs No Pneumonia')
plot_condition_icu_rate_scaled('PREGNANT', ['Not Pregnant', 'Pregnant'], 'Chance of ICU Admission: Pregnancy vs No Pregnancy')
plot_condition_icu_rate_scaled('DIABETES', ['No Diabetes', 'Has Diabetes'], 'Chance of ICU Admission: Diabetes vs No Diabetes')
plot_condition_icu_rate_scaled('INTUBED', ['Not Intubed', 'Intubed'], 'Chance of ICU Admission: Intubed vs Not Intubed')
plot_condition_icu_rate_scaled('COPD', ['No COPD', 'Has COPD'], 'Chance of ICU Admission: COPD vs No COPD')
plot_condition_icu_rate_scaled('INMSUPR', ['Not Immunosuppressed', 'Immunosuppressed'], 'Chance of ICU Admission: Immunosuppressed vs Not Immunosuppressed')
plot_condition_icu_rate_scaled('HIPERTENSION', ['No Hypertension', 'Has Hypertension'], 'Chance of ICU Admission: Hypertension vs No Hypertension')
plot_condition_icu_rate_scaled('CARDIOVASCULAR', ['No Cardiovascular Disease', 'Has Cardiovascular Disease'], 'Chance of ICU Admission: Cardiovascular Disease vs No Cardiovascular Disease')
plot_condition_icu_rate_scaled('RENAL_CHRONIC', ['No Renal Disease', 'Has Renal Disease'], 'Chance of ICU Admission: Renal Disease vs No Renal Disease')
plot_condition_icu_rate_scaled('OTHER_DISEASE', ['No Other Disease', 'Has Other Disease'], 'Chance of ICU Admission: Other Disease vs No Other Disease')
plot_condition_icu_rate_scaled('OBESITY', ['Not Obese', 'Obese'], 'Chance of ICU Admission: Obese vs Not Obese')
plot_condition_icu_rate_scaled('TOBACCO', ['Non-Smoker', 'Smoker'], 'Chance of ICU Admission: Smoker vs Non-Smoker')
plot_condition_icu_rate_scaled('USMER', ['Non-USMER Unit', 'USMER Unit'], 'Chance of ICU Admission: USMER Unit vs Non-USMER Unit')
plot_condition_icu_rate_scaled('SEX', ['Female', 'Male'], 'Chance of ICU Admission: Male vs Female')


def plot_classification_icu_rate_final():

    df_filtered = df.dropna(subset=['CLASIFFICATION_FINAL', 'ICU'])


    classification_counts = df_filtered.groupby('CLASIFFICATION_FINAL')['ICU'] \
                                       .value_counts(normalize=True) \
                                       .unstack().fillna(0) * 100

    if 1.0 in classification_counts.columns:
        icu_only = classification_counts[1.0]
    else:
        icu_only = pd.Series([0] * len(classification_counts), index=classification_counts.index)


    confirmed = icu_only[icu_only.index <= 3]
    inconclusive = icu_only[icu_only.index > 3]
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(confirmed.index.astype(int), confirmed.values, color='#66c2a5', label='Confirmed COVID-19 (1–3)')
    bars2 = plt.bar(inconclusive.index.astype(int), inconclusive.values, color='#fc8d62', label='Inconclusive/Not Carrier (4+)')
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    plt.title('Chance of ICU Admission by COVID Classification')
    plt.xlabel('CLASIFFICATION_FINAL')
    plt.ylabel('Chance of ICU Admission (%)')
    plt.xticks(list(icu_only.index.astype(int)))
    plt.ylim(0, icu_only.max() + 10)
    plt.legend()
    plt.tight_layout()
    plt.show()

#plot_classification_icu_rate_final()

def plot_correlation_matrix():
    df_corr = df.dropna()
    numeric_df = df_corr.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", square=True)
    plt.title("Correlation Matrix (Numeric Features Only)")
    plt.tight_layout()
    plt.show()
#plot_correlation_matrix()


def correlation_matrix_graph():
    df_corr = df.dropna()

    numeric_df = df_corr.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()
    icu_corr = corr_matrix['ICU'].drop('ICU')
    icu_corr_sorted = icu_corr.sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    icu_corr_sorted.plot(kind='bar', color='#5DADE2')
    plt.title('Correlation of Features with ICU Admission')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
#correlation_matrix_graph()
