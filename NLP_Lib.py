from contextlib import redirect_stdout
from genericpath import exists
from multiprocessing import pool
import os
import matplotlib.pyplot
import pandas as pd
import seaborn as sns
import pathlib
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from keras import backend as keras_backend
import tensorflow_hub as hub
from numpy.typing import NDArray
from typing import Tuple, Callable, List
from sklearn.metrics import ConfusionMatrixDisplay
from googletrans import Translator, constants


# import tensorflow.compat.v1 as tf
# disable eager execution to use ELMo model
# tf.disable_eager_execution()

PROJECT_ROOT_DIR = "."
PLOTS_DIR = "plots"
PROJ_COLORS = sns.color_palette("magma")
CLASS_COL = "class"
SUBCLASS_COL = "subclass"
TEXT_COL = "text"
ID_COL = "id"
SUBMISSION_ID_COL = "Id"
SUBMISSION_CATEGORY_COL = "Category"
EMBEDDING_SIZE = 50
PAD_WORD = "--PAD--"
HOMER_SIMPSON = "Homer Simpson"
BART_SIMPSON = "Bart Simpson"
LISA_SIMPSON = "Lisa Simpson"
MARGE_SIMPSON = "Marge Simpson"
NED_FLANDERS = "Ned Flanders"
C_MONTGOMERY_BURNS = "C. Montgomery Burns"
CHIEF_WIGGUM = "Chief Wiggum"
KRUSTY_THE_CLOWN = "Krusty the Clown"
MILHOUSE_VAN_HOUTEN = "Milhouse Van Houten"
GRAMPA_SIMPSON = "Grampa Simpson"
WAYLON_SMITHERS = "Waylon Smithers"
OTHER = "Other"
VALIDATION_SPLIT = "Validation split"
EPOCHS = "Epochs"
STEPS_PER_EPOCH = "Steps per epoch"
NUMBER_OF_TRAINING_STEPS = "Number of training steps"
NUMBER_OF_WARMUP_STEPS = "Number of warmup steps"
INITIAL_LEARNING_RATE = "Initial Learning Rate"
OPTIMIZER = "Optimizer"

CLASSES = [HOMER_SIMPSON, BART_SIMPSON, LISA_SIMPSON, MARGE_SIMPSON, OTHER]
SUBLCASSES = [
    C_MONTGOMERY_BURNS, NED_FLANDERS, CHIEF_WIGGUM, MILHOUSE_VAN_HOUTEN,
    KRUSTY_THE_CLOWN, GRAMPA_SIMPSON, WAYLON_SMITHERS
    ]

ELMO_HANDLE = "https://tfhub.dev/google/elmo/3" # "https://tfhub.dev/google/elmo/2"
# 1st element is the bert preprocessor, the 2nd is the bert encoder.
# For bert encoding, use the electra preprocessor first.
SMALL_BERT = [
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
]

PATH_TO_GENERIC_PLOTS = os.path.join(PROJECT_ROOT_DIR, PLOTS_DIR)
PATH_TO_DATA = os.path.join(PROJECT_ROOT_DIR, "resources")
PATH_TO_TRAINING_DATASET_STRUCT = os.path.join(PATH_TO_DATA, "training")
PATH_TO_SUBCLASS_TRAINING_DATASET_STRUCT = os.path.join(PATH_TO_DATA, "subclass_training")
PATH_TO_TRAINING_TSV = os.path.join(PATH_TO_DATA, "simpsons_dataset-training.tsv")
PATH_TO_TESTING_TSV = os.path.join(PATH_TO_DATA, "simpsons_dataset-testing.tsv")
PATH_TO_MODELS_DIRECTORY = os.path.join(PROJECT_ROOT_DIR, "models")

def save_fig(
    plt: matplotlib.pyplot,
    path_to_fig_and_fig_id: str,
    tight_layout=True,
    fig_extension="png",
    resolution=300) -> None:
    '''
    Given a plot object and save specs/info, saves the plt object's current fig
    to the globally defined plots folder path
    '''
    
    path = f"{path_to_fig_and_fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def make_and_save_inst_per_class_plot(
    df: pd.DataFrame,
    plt: matplotlib.pyplot,
    title: str,
    df_col: str = CLASS_COL):
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


def confusion_matrix(plt, instances_validation, model, path_to_folder) -> None:

    # extract the instances for each class to make a list of lists
    X, y = [], []
    for x, label in instances_validation:
        X.append(x)
        y.append(label)

    # flatten them into just lists
    X = np.concatenate(X, axis=0)    
    y = np.concatenate(y, axis=0)


    predictions = model.predict(X)
    
    # see which predicted class is the same as the given class
    y_hat = []
    for p in predictions:
        y_hat.append([int(item == p.max()) for item in p])
    y_hat = np.array(y_hat)
    
    y = np.transpose(list(map(np.argmax, y)))
    y_hat =  np.transpose(list(map(np.argmax, y_hat)))
        
    acc = np.array(y_hat == y).sum()/len(y)

    plt.clf()
    ConfusionMatrixDisplay.from_predictions(y, y_hat)
    plt.title(f"Confusion Matrix\nAccuracy: {acc}")

    # ensure that the path to save it to exists
    path = os.path.join(path_to_folder, PLOTS_DIR)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_fig(plt, os.path.join(path, "confusion"))

    print("Accuracy:", acc)


def plot_loss(plt: matplotlib.pyplot, history, path_to_folder)-> None:

    # plot loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim((0,4))
    plt.yticks(np.arange(0, 4.5, step=0.5))
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    # ensure that the path to save it to exists
    path = os.path.join(path_to_folder, PLOTS_DIR)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_fig(plt, os.path.join(path, "model_loss"))


def plot_accuracy(plt: matplotlib.pyplot, history, path_to_folder)-> None:

    # plot accuracy
    plt.clf()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.ylim((0,1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')

    # ensure that the path to save it to exists
    path = os.path.join(path_to_folder, PLOTS_DIR)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_fig(plt, os.path.join(path, "model_accuracy"))


def save_training_params(
    validation_split,
    epochs,
    steps_per_epoch,
    num_train_steps,
    num_warmup_steps,
    init_lr,
    optimizer_type,
    path_to_save_at):
    '''
    Saves the given model hyper params to a json at the path given
    '''

    dict = {
        VALIDATION_SPLIT: str(validation_split),
        EPOCHS: str(epochs),
        STEPS_PER_EPOCH: str(steps_per_epoch),
        NUMBER_OF_TRAINING_STEPS: str(num_train_steps),
        NUMBER_OF_WARMUP_STEPS: str(num_warmup_steps),
        INITIAL_LEARNING_RATE: str(init_lr),
        OPTIMIZER: str(optimizer_type)
    }

    with open(os.path.join(path_to_save_at, "training_params.json"), "w") as j:
        j.write(json.dumps(dict, indent=4))


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

def print_model_summary_to_file(model: tf.keras.Model, path_to_model_root: str) -> None:
    '''
    Prints the given model's summary to the given folder.
    '''

    with open(os.path.join(path_to_model_root, "Model_Summary.txt"), 'w') as f:
        with redirect_stdout(f):
            model.summary()


# Function to build model
def build_classifier_model(
    preproc_path: str,
    encoder_path: str,
    optimizer) -> tf.keras.Model:

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(preproc_path, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)

    encoder = hub.KerasLayer(encoder_path, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)

    '''
    Below is a model for predicting subclass
    '''
    embedding = outputs['pooled_output'] # get the resulting BERT encoded/embedded instances
    net = tf.keras.layers.Dense(128, activation="tanh")(embedding)
    net = tf.keras.layers.Dropout(0.55)(net)
    net = tf.keras.layers.Dense(64, activation="tanh")(net)
    net = tf.keras.layers.Dropout(0.35)(net)
    net = tf.keras.layers.Dense(11, activation="softmax", name='classifier')(net)
    
    '''
    Below is the model at ./models/10/16/10:57:45
    embedding = outputs['pooled_output'] # get the resulting BERT encoded/embedded instances
    net = tf.keras.layers.Dense(128, activation="tanh")(embedding)
    net = tf.keras.layers.Dropout(0.55)(net)
    net = tf.keras.layers.Dense(5, activation="softmax", name='classifier')(net)
    '''
    '''
    Below is the model at ./models/10/18/20:10:14/
    
    embedding = outputs['sequence_output'] # get the resulting BERT encoded/embedded instances
    # Conv1D for temporal data
    net = tf.keras.layers.Conv1D(512, 2, activation="tanh", use_bias=False, padding="same")(embedding)
    net = tf.keras.layers.Conv1D(256, 3, activation="tanh", use_bias=False, padding="same")(net)
    net = tf.keras.layers.Conv1D(256, 4, activation="tanh", use_bias=False, padding="same")(net)

    net = tf.keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.2)(net)
    net = tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.35)(net)

    # pool and flatten into Dense layer form
    net = tf.keras.layers.MaxPool1D(128)(net)
    net = tf.keras.layers.Flatten()(net)

    net = tf.keras.layers.Dense(128, activation="tanh")(net)
    net = tf.keras.layers.Dropout(0.44)(net)
    net = tf.keras.layers.Dense(32, activation="tanh")(net)
    net = tf.keras.layers.Dropout(0.55)(net)
    net = tf.keras.layers.Dense(5, activation="softmax", name='classifier')(net)
    '''

    '''
    Below is the model at ./models/10/18/16:29:49/
    embedding = outputs['sequence_output'] # get the resulting BERT encoded/embedded instances
    # Conv1D for temporal data
    net = tf.keras.layers.Conv1D(512, 2, activation="tanh", padding="same")(embedding)
    net = tf.keras.layers.Conv1D(256, 2, activation="tanh", padding="same")(net)
    net = tf.keras.layers.Conv1D(128, 2, activation="tanh", padding="same")(net)

    # pool and flatten into Dense layer form
    net = tf.keras.layers.MaxPool1D(128)(net)
    net = tf.keras.layers.Flatten()(net)

    net = tf.keras.layers.Dense(128, activation="tanh")(net)
    net = tf.keras.layers.Dropout(0.4)(net)
    net = tf.keras.layers.Dense(32, activation="tanh")(net)
    net = tf.keras.layers.Dense(5, activation="softmax", name='classifier')(net)
    '''

    model = tf.keras.Model(text_input, net)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=tf.metrics.CategoricalAccuracy())

    '''
    Layers scratch
    # net = tf.keras.layers.Conv1D(128, 3, activation="tanh", padding="same")(pool1)
    # net = tf.keras.layers.Flatten()(tf.keras.layers.concatenate([pool1, pool2, pool3]))
    # tf.keras.layers.Reshape(target_shape, **kwargs)
    '''

    return model


def delete_instances_from_file_struct(
    sub_string_to_del_if: str,
    path_to_root_of_file_struct: str):
    '''
    Deletes files in a dataset file/folder struct if the file name contains the given
    "sub_string_to_del_if" in its name. Used at least to delete the accidental copies
    of instances we made by converting the instances to english, then back to english.
    '''

    for (_, dirnames, _) in os.walk(path_to_root_of_file_struct):
        for d in dirnames:
            for (_, _, inst_filenames) in os.walk(os.path.join(path_to_root_of_file_struct, d)):
                for f in inst_filenames:
                    if sub_string_to_del_if in f:
                        print(f"deleted {f}")
                        os.remove(os.path.join(path_to_root_of_file_struct, d, f))


def translate_df_to_and_back(
    dest_lang: str,
    training_df: pd.DataFrame,
    fraction_sample_size_based_on_class: Callable[[str], float], # is a function of the class string given
    classes: list,
    class_col: str) -> pd.DataFrame:
    '''
    Takes a dataframe in the form of the training data .tsv, a language to translate that
    dataframe's text column to, does the translation to that language, then translates it back
    to english. The translations can take a long time, so a fraction value is taken as well and
    a random sample of that size is taken from each class translated.
    fraction_of_each_class_to_translate = 0.001 does not take forever and takes about 100
    random ones spread about the classes and translates them.
    '''

    translator = Translator()
    sample_of_translated = pd.DataFrame(columns=training_df.columns)
    for c in classes:
        class_c_instances = training_df[training_df[class_col] == c].sample(
            frac=fraction_sample_size_based_on_class(c),
            replace=False)    
        # translate from english to dest_lang, then translate back, from dest_lang to english
        class_c_instances[TEXT_COL] = class_c_instances[TEXT_COL].map(lambda x: translate_str_to_and_back(translator, x, dest_lang))
        sample_of_translated = pd.concat([sample_of_translated, class_c_instances])
   
    return sample_of_translated


def translate_str_to_and_back(translator, text_to_translate: str, dest_lang: str) -> str:

    '''
    Uses the given translator to translate the given string into the given langauge,
    "dest_lang", then back to english, then returns it.
    '''

    return translator.translate(
        translator.translate(text_to_translate, dest=dest_lang, src="en").text,
        dest="en",
        src=dest_lang).text


def increase_training_data_via_language_traslation(
    training_df: pd.DataFrame,
    fraction_sample_size_based_on_class: Callable[[str], float], # is a function of the class string given
    classes_to_make_more_instances_of: list,
    path_to_root_of_file_ds_struct: str,
    class_column) -> None:
    '''
    Takes a dataframe in the form of the training data .tsv and translates a random sample
    of size (fraction_sample_size * total instances of class) for each class into every
    language provided by google translate, then translates it back and saves it as a .txt
    in the file/folder data struct whose root directory is "path_to_root_of_file_ds_struct".
    '''

    for key in constants.LANGUAGES:
        try:
            # dont translate english to english then back to enlgish
            if key != "en":
                lang_i = constants.LANGUAGES[key]
                print(lang_i) # print the language it is starting to show progression
                translated_df = translate_df_to_and_back(
                    key,
                    training_df,
                    fraction_sample_size_based_on_class,
                    classes_to_make_more_instances_of,
                    class_column)
                make_df_into_ds_file_form(
                    translated_df,
                    path_to_root_of_file_ds_struct,
                    classes_to_make_more_instances_of,
                    class_column,
                    f"{lang_i}_")
        except Exception:
            print("Error on language: ", lang_i)


def get_translation_frac_size(cls: str) -> float:
    '''
    Used to determine how much of a sample size to take from each class when doing
    translating to other languages and back. Using different sample sizes so that
    the number of instances for each class can become more even.
    '''

    if cls == HOMER_SIMPSON:
        return 0.0240
    elif cls in CLASSES:
        return 0.0525
    else:
        return 0.1550

def map_probs_to_class(probs: list[list[float]]) -> pd.Series:
    '''
    Given a probability vector, returns the class that it represents
    with regards to the model/weights we have made. If the model's
    prediction is [0.2, 0.6, 0.4, 0.02, 0.09], then that is the
    probability of
    [Bart Simpson, Homer Simpson, Lisa Simpson, Marge Simpson, Other].
    From that array, 0.6 is the highest probability, so return Homer Simpson.
    for that instance. Do that for all instances.
    '''

    y_hat = []
    for p in probs:
        y_hat.append(
            map_idx_to_class(np.argmax([int(item == p.max()) for item in p])))
    
    return pd.Series(y_hat)


def map_idx_to_class(idx: int) -> str:
    '''
    Given an idx, return the corresponding class. This is dependent on the way
    our model has represented the classes in a softmax vector.
    '''

    if idx == 0:
        return BART_SIMPSON
    elif idx == 1:
        return HOMER_SIMPSON
    elif idx == 2:
        return LISA_SIMPSON
    elif idx == 3:
        return MARGE_SIMPSON
    else:
        return OTHER


def map_idx_to_class(idx: int) -> str:
    '''
    Given an idx, return the corresponding sub class. This is dependent on the way
    our model has represented the classes in a softmax vector.
    '''

    if idx == 0:
        return BART_SIMPSON
    elif idx == 1:
        return C_MONTGOMERY_BURNS
    elif idx == 2:
        return CHIEF_WIGGUM
    elif idx == 3:
        return GRAMPA_SIMPSON
    elif idx == 4:
        return HOMER_SIMPSON
    elif idx == 5:
        return KRUSTY_THE_CLOWN
    elif idx == 6:
        return LISA_SIMPSON
    elif idx == 7:
        return MARGE_SIMPSON
    elif idx == 8:
        return MILHOUSE_VAN_HOUTEN
    elif idx == 9:
        return NED_FLANDERS
    elif idx == 10:
        return WAYLON_SMITHERS
    else:
        raise Exception("Invalid class index!")


def make_df_into_ds_file_form(
    training_df: pd.DataFrame,
    path_to_file_struct_root: str,
    classes: list,
    class_col_to_use: str,
    unique_id: str = ""):

    '''
    Takes a dataframe representing a .tsv in the form of the training data .tsv,
    a path to root folder for a file/folder data set structure to be made at,
    a list of target classes to turn to folders and add the corresponding instances
    into, and a unique_id to prefix file names (which are coded to just be the row ids
    of the given dataframe, not the row index, but the row id) so that the same file names
    can be used to save multiple instances, but with slightly different prefixes. For
    instance, training instance 112, being a Bart class, the text being "Eat my shorts",
    can be saved twice as path_to_file_struct_root/Bart/es112.txt and
    path_to_file_struct_root/Bart/en112.txt, where the "es" one is "Eat my shorts"
    translated to spanish, then back to english, and saved to as a .txt as what ever the
    translation to and back becomes, and the "en" one is just the original english version,
    "Eat my shorts" saved as a .txt.

    The file/folder data set structure that is made is (and must be) similar to the form:
    ./some_path/to_training_ds/
                            ___Bart/
                            _______bart_instance_id_r.txt
                            _______bart_instance_id_q.txt
                            _______ ...
                            ___Homer/
                            ________homer_instance_id_k.txt
                            ________homer_instance_id_n.txt
                            ________ ...
                            ___Marge/
                                    ......
                            ___Lisa/
                                    ......
                            ___Other/
                            ________other_instance_id_k.txt
                            ________other_instance_id_n.txt
                            ________ ...
    
    where the folders "Bart". "Homer", ..., "Other" are named after the class that each
    folder represents and the instances inside each folder are instances of that class.
 
    '''

    # if unique_id has any / or \ in it, it will create new folders and mess up the
    # dataset structure
    assert "\\" not in unique_id and "/" not in unique_id
    for c in classes:
        class_c_path = f"{path_to_file_struct_root}/{c}"
        pathlib.Path(class_c_path).mkdir(parents=True, exist_ok=True)
        class_c_instances = training_df[training_df[class_col_to_use] == c].filter(items=[ID_COL, TEXT_COL])
        for row in class_c_instances.iterrows():
            series = row[1]
            with open(f"{class_c_path}/{unique_id}{series[ID_COL]}.txt", "w") as file:
                file.write(series[TEXT_COL])





# Not used, but Elmo object taken from 
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


# *** SCRATCH ***
'''
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


# create an instance of ELMo
embeddings = language_model(X, signature="default", as_dict=True)["elmo"]
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# or a layer of ELMo
hub.KerasLayer( # This layer is a word embedding layer
        language_model_path,
        trainable=False
        #dtype=tf.string,
        #output_key="elmo",
        #input_shape=(1,),
        #output_shape=[output_shape_size]
        ),

BUILD EMBEDDER MANUALLY
max_seq_len = training_df[TEXT_COL].str.len().max()

preprocessor = hub.KerasLayer(SMALL_BERT[0])

encoder_inputs = preprocessor(training_df[TEXT_COL])
encoder = hub.KerasLayer(SMALL_BERT[1], trainable=True)


BUILD X, y for training and validation MANUALLY
embedded = encoder_outputs["default"]
Y = tf.keras.utils.to_categorical(classes_encoding)
eighty_percent = int(embedded.shape[0] * 0.8)
classes_encoding = training_df[CLASS_COL].map(map_class_to_float)
X_train = embedded[0:eighty_percent]
Y_train = tf.keras.utils.to_categorical(classes_encoding[0:eighty_percent])
X_validate = embedded[eighty_percent:]
Y_validate = tf.keras.utils.to_categorical(classes_encoding[eighty_percent:])
'''