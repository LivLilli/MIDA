# MIDA
Multiple Imputation using Denoising Autoencoders


# DUBBI:

* normalize or standardize train and test (?); at the end, inverse scale?

* target data: pulito con i na riempiti perchè la rete non legge i na ed ha il compito di ricostruire un dataset completo.

* early stoppping: if simple moving average of length 5 of the error deviance does not improve (?).

* algoritmo multiple impu: 5 imputazioni su uno stesso dataset, con diversi pesi inizializzati random -> col seed sono sempre uguali. allora rimuovi seed iniziale e mettilo dentro il for i in range(5): set_seed(i).

* batch_size = 500 -> più veloce!
