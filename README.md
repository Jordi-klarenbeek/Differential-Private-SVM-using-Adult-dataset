# Differential Private SVM using Adult Dataset
This SVM is created to classify the Adult dataset, with the requirements of differential privacy. Differential privacy is a system for protecting the anonymity of individuals in a dataset by adding noise to the data or ML algorithm.
Differential privacy prevents an attacker from infering information about an individual from publicly available datasets.

This implementiation was based on the paper of Chaudhuri et al (2011), using the objective perturbation approach of algorithm 2 of the paper.
The dataset mainly used for training and evaluating the differential private SVM model was the Adult dataset from the UCI dataset repository (Kohavi & Becker, 1996).

Main file to run is the adult.py file, in this file the data is prepared and the SVM is trained by calling testing.py.
Epsilon is a measure for how strictly private the data needs to be, so run_epsilon in testing.py is used to compare the trained svm of multiple epsilon values. 
Other functions in testing.py were used to train the hyperparameters of the svm model. Reason for not having a main.py file to run the whole program was that the model was also trained for other datasets, which had their own run file.

# References
1. Chaudhuri, K., Monteleoni, C., & Sarwate, A. D. (2011). Differentially private empirical risk minimization. Journal of Machine Learning Research, 12(3).
2. Kohavi R. & Becker B. (1996). Adult Data Set [Data set]. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/adult
