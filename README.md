# CS 567 NLP

## Example of setting up a conda environment for this project
#### Maybe not neccesary, but guarantees you'll be able to use, debug, run the code + keep you from creating a clashing Python version dependency between Python projects. Have conda installed, then in a conda terminal:
* `conda create -n NLP python=3.10`
* `conda activate NLP`
* `pip install tensorflow googletrans==3.1.0a0 tensorflow-text datasets tf-models-official eyeD3 jupyterlab numpy matplotlib pandas scikit-learn colormap seaborn`

Now, when you work in the project and/or run its code, do so in this NLP environment/conda space.

If you are attempting to train a tensorflow NN and are getting a warning about the work not being
mapped to the GPU because the library `libcudnn8` is not installed, then you can install it via conda:
* `conda install -c anaconda cudnn`

## Files
* `NLP_Lib.py` -- This is a python library of functions we defined to help with cleaning, processing data, etc.
* `NLP_Training.ipynb` -- This is a notebook for processing data, conducting NLP experiments, producing plots, analyzing the data, etc.
* `NLP_Classification.ipynb` -- This is a notebook for loading in saved model specs and making prediction files.
* `NLP2.ipynb` -- This is a notebook that use EMLo on a small subset and preformed very poorly. It is not in a working state, but has some evidence of our experiments.
* `Pytorch.ipynb` --  This is a notebook that we try to switch pytorch and we found it very diffecutly to use, the train model was realy hit or miss. It is not in a working state, but has some evidence of our experiments.

## General setup before running any code
1. Ensure the testing and training data csv files are in the folder `project_root_directory/resources/`

## Using the NLP code
- You can find some nice comments and explainations throughout the code, in both the notebook files and amongst the functions in `NLP_Lib.py`.
- The file `NLP_Training.ipynb` has the work flow for training models, along with comments explaining how to use it.
    1. First you must construct a file/folder data structure out of your data using the function `make_df_into_ds_file_form` in `NLP_Lib.py`.
    2. Load in the data from the file/floder struct you made as a Dataset object.
    3. Define your model parameters in the notebook and your architecture in the function `buiild_classifier_model` in `NLP_Lib.py`.
    4. During training, model checkpoints will be made at each epoch that you load in during classification.
    5. The last notebook block in the training workflow will save the model's hyper params, the accuracy and loss plots, and the architecture to the current model directory, defined by a date and time.
- The file `NLP_Classification.ipynb` has the work flow for classifying with a saved model, along with comments explaining how to use it.
    1. In the second and third notebook block, load in the specs of the model you want
        - use the plots saved during training of the model you trained to decided on a good epoch weights to load in.
        - load in the hyper params from the model's "training_params.json" saved at the end of training so that the model you build matches the one you trained.
        - make sure the architecture of the model returned by the function `buiild_classifier_model` in `NLP_Lib.py` is close enough to the one saved in your chosen model's directory as `Model_Summary.txt` during training.
    2. In the final few blocks of the notebook, make predictions with your loaded in model, then use them to make a submission file.
        - Depending on how your model discriminated, you may need to map your predictions from subclass predictions to class predictions: "Ned Flanders" -> "Other".
- The file `NLP_Lib.py`
    - has functions to use during the training and classification workflow, with comments for use.
    - has many globally defined variables to make switching between training/classifying with different setups easier.
