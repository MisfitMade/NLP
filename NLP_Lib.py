import os
import matplotlib.pyplot
import pandas as pd
import seaborn as sns
import pathlib
import numpy as np
from tensorflow import keras
from keras import backend as keras_backend
import tensorflow_hub as hub
from numpy.typing import NDArray
from typing import Tuple, Callable, List
from sklearn.metrics import ConfusionMatrixDisplay


import tensorflow.compat.v1 as tf
# disable eager execution to use ELMo model
tf.disable_eager_execution()

PROJECT_ROOT_DIR = "."
PATH_TO_PLOTS = f"{PROJECT_ROOT_DIR}/plots/"
PATH_TO_DATA = f"{PROJECT_ROOT_DIR}/resources"
PATH_TO_TRAINING_DATA = f"{PATH_TO_DATA}/simpsons_dataset-training.tsv"
PATH_TO_PADDED_X = f"{PATH_TO_DATA}/padded_x.npz"
CHECKPOINT_DIR = f"{PROJECT_ROOT_DIR}/model_checkpoints/"
PROJ_COLORS = sns.color_palette("magma")
ELMO_HANDLE = "https://tfhub.dev/google/elmo/3" # "https://tfhub.dev/google/elmo/2"
CLASS_COL = "class"
SUBCLASS_COL = "subclass"
TEXT_COL = "text"
EMBEDDING_SIZE = 50
PAD_WORD = "--PAD--"
HOMER_SIMPSON = "Homer Simpson"
BART_SIMPSON = "Bart Simpson"
LISA_SIMPSON = "Lisa Simpson"
MARGE_SIMPSON = "Marge Simpson"

# Ensure the existence of certain folders
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


def make_and_save_inst_per_class_plot(
    df: pd.DataFrame,
    plt: matplotlib.pyplot,
    title: str,
    df_col: str = "class"):
    '''
    Takes the training data dataframe and plots the number of instances per class,
    saves the figure too.
    '''

    pd.Series(df[df_col]).value_counts().plot(
        kind = "bar",
        color=PROJ_COLORS,
        title=title)
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
            f"*****Training*****\n{training[TEXT_COL]}\n*****Validation*****\n{validation[TEXT_COL]}\n")

    return training.filter(items=[TEXT_COL, CLASS_COL]), validation.filter(items=[TEXT_COL, CLASS_COL])


def confusion_matrix(instances_validation, model, plt, fig_id) -> None:

    # confusion matrix
    X, y = [], []
    for x, label in instances_validation:
        X.append(x)
        y.append(label)

    X = np.concatenate(X, axis=0)    
    y = np.concatenate(y, axis=0)

    predictions = model.predict(X)
    y_hat = predictions.argmax(axis=-1)

    acc = (y == y_hat).sum() / len(y)

    ConfusionMatrixDisplay.from_predictions(y, y_hat)
    plt.title(f"Confusion Matrix\nAccuracy: {acc}")
    save_fig(plt, f"{fig_id}_confusion")

    print("Accuracy:", (y == y_hat).sum() / len(y))



def plot_loss(plt: matplotlib.pyplot, history, fig_id)-> None:

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim((0,4))
    plt.yticks(np.arange(0, 4.5, step=0.5))
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    save_fig(plt, f"{fig_id}_model_loss")


def plot_accuracy(plt: matplotlib.pyplot, history, fig_id)-> None:

    # plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylim((0,1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    save_fig(plt, f"{fig_id}_model_accuracy")


def map_class_to_float(classification: str) -> float:

    if classification == HOMER_SIMPSON:
        return 0
    elif classification == BART_SIMPSON:
        return 1
    elif classification == LISA_SIMPSON:
        return 2
    elif classification == MARGE_SIMPSON:
        return 3
    else:
        return 4

def map_str_to_padded_longest_str(str: str, longest_str_ln: int) -> str:
    '''
    Adds --PAD-- to a seqeunce of words if it is not 'longest_str_ln' number
    of words, until it is 'longest_str_ln' number of words.
    '''

    words_list = str.split() # split on spaces
    num_words = len(words_list)
    while num_words < longest_str_ln:
        str += f" {PAD_WORD}"
    
    return str
    

def get_X(text_column: pd.Series, need_fresh_padded_x = False):

    
    if need_fresh_padded_x:
        max_str_len = text_column.str.len().max()

        padded_x = text_column.map(
            lambda s: map_str_to_padded_longest_str(s, max_str_len))
        # np.savez(PATH_TO_PADDED_X, padded_x)

        return padded_x
    
    # else
    return np.load(PATH_TO_PADDED_X)


# Elmo object taken from 
# https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
class ElmoEmbeddingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module(
            ELMO_HANDLE,
            trainable=self.trainable,
            name="{}_module".format(self.name))

        self.trainable_weights += keras_backend.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            keras_backend.squeeze(keras_backend.cast(x, tf.string), axis=1),
            as_dict=True,
            signature='default',
            )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return keras_backend.not_equal(inputs, PAD_WORD)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# Function to build model
def build_model(act_fnx: str = "tanh", output_act_fnx: str = "softmax"): 
    input_text = keras.layers.Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = keras.layers.Dense(256, activation=act_fnx)(embedding)
    pred = keras.layers.Dense(1, activation=output_act_fnx)(dense)

    model = keras.models.Model(inputs=[input_text], outputs=pred)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), # SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax Nadam, Ftrl
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    model.summary()
    
    return model




# *** SCRATCH ***
'''
hub.KerasLayer( # This layer is a word embedding layer
        language_model_path,
        trainable=False
        #dtype=tf.string,
        #output_key="elmo",
        #input_shape=(1,),
        #output_shape=[output_shape_size]
        ),

# Now lets define our NN, using ELMo as the first layer, the embedding layer
output_shape_size = 150
language_model_path = ELMO_HANDLE
activation_fnx = "tanh"

model = keras.Sequential([
    ElmoEmbeddingLayer()(keras.layers.Input(dtype='string', input_shape=(1,))),
    keras.layers.Dense(128, activation=activation_fnx),
    keras.layers.Dense(5, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(), # SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax Nadam, Ftrl
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"])

model.summary()

# sharding settings for dev defined GPU work distribution
# (AutoShardPolicy.DATA or AutoShardPolicy.OFF)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA



# create an instance of ELMo
embeddings = language_model(X, signature="default", as_dict=True)["elmo"]
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
'''