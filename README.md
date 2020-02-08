# MIDA
Multiple Imputation using Denoising Autoencoders


# DUBBI:

* normalize or standardize train and test (?); at the end, inverse scale?

* early stoppping: if simple moving average of length 5 of the error deviance does not improve (?);

* mean rmse simile per df corrotti aventi medesimo df target. 

* 60 df, after modeling, do'nt have na vvalues.

* set seed generale, seed per vettore, seed per split train-test, seed per pesi...

* prova adam: fix della conversione numpy-tensor.

* batch size.

* file per tensorboard non sovrascrive: cancella e rigira.
