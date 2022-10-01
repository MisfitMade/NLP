import os
import matplotlib.pyplot

PROJECT_ROOT_DIR = "."
PATH_TO_PLOTS = f"{PROJECT_ROOT_DIR}/plots"

def save_fig(plt: matplotlib.pyplot, fig_id: str, tight_layout=True, fig_extension="png", resolution=300) -> None:
    '''
    Given a plot object and save specs/info, saves the plt object's current fig
    to the globally defined plots folder path
    '''
    
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    path = os.path.join(PATH_TO_PLOTS, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)