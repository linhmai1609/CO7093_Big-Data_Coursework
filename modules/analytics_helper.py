import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns


def plot_multiple_metrics(df: pd.DataFrame, columns: list, figsize: tuple):
    # Set up a grid of plots
    fig = plt.figure(figsize=(20,20)) 
    fig_dims = (math.ceil(len(columns)/4), 4)
    counter = 0

    for i in range(math.ceil(len(columns)/4)):
        for j in range(4):
            if counter < len(columns):
                plt.subplot2grid(fig_dims, (i, j))
                df[columns[counter]].value_counts(dropna=False).sort_index().plot(kind='bar', 
                                                                            title=columns[counter])
                counter += 1