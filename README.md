# RoBoTS: A Robust Bounded Twin SVM Based on RoBoSS Loss Function

Title: RoBoTS: A Robust Bounded Twin SVM Based on RoBoSS Loss Function
Authors: Mushir Akhtar, M. Tanveer, Mohd. Arshad
Status: Under revision in Pattern Recognition (Elsevier)

============================================================
README
============================================================

This repository provides the MATLAB implementation of the proposed
RoBoTS (Robust Bounded Twin Support Vector Machine) based on the
RoBoSS loss function for nonlinear binary classification.


------------------------------------------------------------
Files in this repository
------------------------------------------------------------

1) Main_RoBoSS_TSVM.m

   Main script for RoBoTS (nonlinear kernel version).
   This script:
   - loads a dataset from a .txt or .mat file,
   - converts labels from {1,0} to {1,-1} when needed,
   - performs a stratified hold-out split (75% training, 25% testing),
   - normalizes the data using training statistics,
   - trains the RoBoTS model,
   - reports the test accuracy.

   This file is designed as a demo to directly run the
   algorithm on the provided example dataset.


2) RoBoSS_TSVM_function.m

   Core implementation of the proposed RoBoTS learning algorithm.

   This function:
   - constructs nonlinear kernel representations,
   - separates positive and negative class samples,
   - optimizes the two bounded twin hypersurfaces using the proposed
     RoBoSS loss formulation,
   - estimates the corresponding model parameters,
   - performs prediction using the distance to the two hypersurfaces.

   The function returns the learned model parameters, test accuracy,
   and training time.


3) kernelfunction.m

   Utility function to compute kernel values.
   The current implementation supports:
   - linear kernel,
   - polynomial kernel,
   - radial basis function (RBF) kernel.

   In the provided experiments, the RBF kernel is used.


------------------------------------------------------------
Experimental setup
------------------------------------------------------------

In the experimental study reported in the paper, model performance is assessed using a 4-fold cross-validation strategy, where in each fold, 75% of the samples are used for training and the remaining 25% for testing. For every hyperparameter configuration, training and testing are performed across all folds, and the highest testing accuracy obtained among them is reported as the final result for each dataset.


------------------------------------------------------------
Hyperparameter ranges (used in the full experimental study)
------------------------------------------------------------

In the full experimental evaluation reported in the paper, the hyperparameters were selected using grid search with cross-validation.

The tuning ranges are:

C     = 10.^(-6 : 2 : 6)     (structural regularization parameter)
c     = 10.^(-6 : 2 : 6)     (loss regularization parameter)
sigma = 10.^(-6 : 2 : 6)     (RBF kernel width)
a     = 0.1 : 0.2 : 5.1      (RoBoSS loss parameter)
b     = 0.5 : 0.5 : 1.5      (RoBoSS loss parameter)

Four-fold cross-validation was used for hyperparameter selection.


------------------------------------------------------------
Input data format
------------------------------------------------------------

The code supports the following input formats:

1) .txt file
   A numeric matrix where the last column represents the class label.

2) .mat file
   Either:
   - a single matrix variable whose last column is the label, or
   - variables X and y.

The classification task must be binary.
Labels are expected to be in {1,0} or {1,-1}.
If labels are given as {1,0}, they are internally converted to {1,-1}.


------------------------------------------------------------
How to run
------------------------------------------------------------

Place the following files in the same directory:

- Main_RoBoSS_TSVM.m
- RoBoSS_TSVM_function.m
- kernelfunction.m
- congressional_voting.mat   (demo dataset)

Then simply run in MATLAB:

>> Main_RoBoSS_TSVM


------------------------------------------------------------
Remarks
------------------------------------------------------------

This repository provides a simplified demo version for easy usage and
verification. The full experimental pipeline including exhaustive
hyperparameter tuning and cross-validation is described in the paper.


------------------------------------------------------------
Contact
------------------------------------------------------------

For any queries, please contact:

phd2101241004@iiiti.ac.in

