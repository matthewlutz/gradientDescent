Mini-Batch Gradient Descent Program

Prerequisites
-------------
JDK installed, or eclipse installed

How to Run
----------
Compile the Program:
--------------------
javac Main.java

Use the following command to execute the program from the command line:
-----------------------------------------------------------------------
java Main -f <filename> [-k <k-fold>] [-d <min_degree>] [-D <max_degree>] [-a <alpha>] [-e <epoch_limit>] [-m <batch_size>] [-r] [-v <verbosity>]

Command Line Arguments
----------------------
-f <filename>: Specifies the path to the input data file (required).
-k <k-fold>: Specifies the number of folds for cross-validation (default is 1).
-d <min_degree>: Specifies the smallest polynomial degree to evaluate (default is 1).
-D <max_degree>: Specifies the largest polynomial degree to evaluate (default is 1).
-a <alpha>: Sets the learning rate (default is 0.005).
-e <epoch_limit>: Sets the maximum number of epochs for training (default is 10000).
-m <batch_size>: Specifies the size of each mini-batch (default is 0, which uses the entire dataset).
-r: Randomizes the order of data points before training.
-v <verbosity>: Sets the level of output verbosity (default is 1, with levels ranging from 1 to 5).


Not finished
-------------
Degree specification is sometimes buggy
Cross validation output only works for the basic output 