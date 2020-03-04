# MIDA
## Multiple Imputation using Denoising Autoencoders

This project consists in the implementation of experiments explained in the above mentioned paper. In particular, the authors built a denoising autoencoder which, given a corrupted dataset, is able to recover the actual one, with the implementation of a multiple imputation. The several experiments are based on different kinds of dataset corruptions and are then compared with results coming from the SOTA (state of the art) method, MICE. 



#   Repository Description

>* *load_data.R* : R code for downloading datasets.

>* *mice.R* : R code for the implementation of the state of the art (MICE).

>* *missingness.py* : Python class for adding corruption to data.

>* *model.py* : Python classes for building, training and evaluating the DAE model.

>* *preprocessing.py* : Python class for preprocess data.

>* *project.ipynb* : official tf implementation and report. 

>* *requirements.py* : requirements for Python code.

>* *tensorboard.ipynb* : tf tensorboard for train and test loss representation. 

>* *results* folder: it contains two csv files with results of DAE and MICE models.

>* *saved_models* folder: it contains trained DAE models saved.

>* *data* folder: it contains csv file of the 15 original datasets.

>* *tensorboards* folder: it contains the files for plotting losses on the Tensorboard.

>* *corrupted_datasets* folder: it contains csv files of the 60 corrupted datasets.
