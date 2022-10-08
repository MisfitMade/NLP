import os
import matplotlib.pyplot
import pandas as pd
import seaborn as sns

PROJECT_ROOT_DIR = "."
PATH_TO_PLOTS = f"{PROJECT_ROOT_DIR}/plots/"
PATH_TO_TRAINING_DATA = f"{PROJECT_ROOT_DIR}/resources/simpsons_dataset-training.tsv"
PROJ_COLORS = sns.color_palette("magma")

def save_fig(
    plt: matplotlib.pyplot,
    fig_id: str,
    tight_layout=True,
    fig_extension="png",
    resolution=300) -> None:
    '''
    Given a plot object and save specs/info, saves the plt object's current fig
    to the globally defined plots folder path
    '''
    
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    path = os.path.join(PATH_TO_PLOTS, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def make_and_save_inst_per_class_plot(df: pd.DataFrame, plt: matplotlib.pyplot, df_col: str = "class"):
    '''
    Takes the training data dataframe and plots the number of instances per class,
    saves the figure too.
    '''

    pd.Series(df[df_col]).value_counts().plot(
        kind = "bar",
        color=PROJ_COLORS,
        title=f"Number of Instances in Each {df_col}")
    plt.xlabel('Characters')
    plt.ylabel('No. of Instances')
    save_fig(plt, f"instances_per_{df_col}", True)