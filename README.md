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

## General setup before running any code

## Using the NLP code
- The file `NLP_Training.ipynb`
    - has the work flow for training models, along with comments explaining how to use it.
- The file `NLP_Classification.ipynb`
    - has the work flow for classifying with a saved model, along with comments explaining how to use it.
- The file `NLP_Lib.py`
    - has functions to use during the training and classification workflow, with comments for use.
    - has many globally defined variables to make switching between training/classifying with different setups easier.