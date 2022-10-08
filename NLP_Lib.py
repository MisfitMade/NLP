import os
import matplotlib.pyplot
import pandas as pd
import seaborn as sns
import pathlib
from numpy.typing import NDArray
from typing import Tuple, Callable, List

PROJECT_ROOT_DIR = "."
PATH_TO_PLOTS = f"{PROJECT_ROOT_DIR}/plots/"
PATH_TO_TRAINING_DATA = f"{PROJECT_ROOT_DIR}/resources/simpsons_dataset-training.tsv"
CHECKPOINT_DIR = f"{PROJECT_ROOT_DIR}/model_checkpoints/"
PROJ_COLORS = sns.color_palette("magma")
ELMO_HANDLE = "https://tfhub.dev/google/elmo/2"
CLASS_COL = "class"
SUBCLASS_COL = "subclass"
TEXT_COL = "text"

# Ensure the existence of certain folders
pathlib.Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(PATH_TO_PLOTS).mkdir(parents=True, exist_ok=True)

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


def train_valid_split(
    training_df: pd.DataFrame,
    window_start: int,
    window_end_idx_plus1: int,
    report: bool = False) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

    training = training_df.iloc[:window_start,:]
    validation = training_df.iloc[window_start:window_end_idx_plus1,:]
    training = pd.concat([training, training_df.iloc[window_end_idx_plus1:,:]])

    if report:
        print(
            f"*****Training*****\n{training}\n*****Validation*****\n{validation}\n")

    return training[TEXT_COL], validation[TEXT_COL], training[CLASS_COL], validation[CLASS_COL]