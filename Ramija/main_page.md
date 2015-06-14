# Analysis of the data set

The data set includes 93 attributes. Additionally there is another attribute which stores the class variables. Those variables can be assigned to one of possible nine categories. Overall the data set has more than 200.000 instances. 140.000 instances are unlabeled and used for the test set whereas the 60.000 remaining labeled instances are used as training set. The data set itself doesn't contain any missing value, so no cleaning is required.


# Select an appropriate subset of the attributes

We use the principal component analysis method (PCA) to reduce the dependency between attributes. This method uses linear combinations to find the dependency. If the search was successful, the dependent attributes will be merged together to 1 attribute. This means the dimension and the number of dependencies will be reduced which improves the analysis of the data.
We used the PCA filter of weka for the PCA method for both, the training set and the test set. Both sets will get the same attributes. Weka itself is a collection of machine learning algorithms for data mining.

# Used Machine Learning algorithms

## Semi Supervised Learning:

Both labeled and unlabeled data is used of the learners for the semi supervised learning.
We implemented a self-training approach with k-nearest neighbors, namely the IBk-algorithm of weka. A random element will be chosen and classified. Afterwards it will be added to the labeled data.
The main problem with other, pre-implemented algorithms was a very high memory usage, that's why we chose this simple solution.

## SVM Classifier:

SVM, or Support Vector Machine, separates a set of data with a hyperplane into two clusters. The margin to the hyperplane should be maximized. If there are more than two classes, this approach has to be used multiple times to get a proper result.
We used the weka library libsvm for this classifier. To evaluate our result we split our labeled training set into two equally large sets and use one subset for evaluation. Two-fold cross-validation and multi-class logarithmic loss were used to evaluate our model. The results are written below.


## J48/C4.5 Classifier:

This classifier is based on a decision tree. This tree is build up with training data and should classify other instances. There is a weka library called j48 which includes functions for the C4.5 classifier. We used this to create our model and evaluated it with two-fold-cross-validation and multi-class logarithmic loss too.


# Evaluation of our model

This chapter includes an evaluation for our two classifiers. We cannot evaluate our semi supervised approach due to the lack of labeled data within our test set. Additionally the program takes a long time to work. 


## Results of SVM classifier


Correctly Classified Instances       	49986               80.7815 %
Incorrectly Classified Instances     	11892               19.2185 %
Kappa statistic                          		0.764
Mean absolute error                     	0.0427
Root mean squared error                 	0.2067
Relative absolute error                 	23.1224 %
Root relative squared error            	68.0039 %
Total Number of Instances            	61878     

multi-class logarithmic loss of SVM model is: 6.63782166340675

## Results of decision tree classifier


Correctly Classified Instances       	59290              95.8176 %
Incorrectly Classified Instances      	2588                4.1824 %
Kappa statistic                          		0.9496
Mean absolute error                      	0.0145
Root mean squared error                  	0.0852
Relative absolute error                  	7.8681 %
Root relative squared error             	28.0503 %
Total Number of Instances            	61878     

multi-class logarithmic loss of decision tree is: 0.12237539427354557

As you can see, both algorithms have with more than 80% a high amount of correct classified instances. C4.5 classifier (named decision tree classifier above) however has almost 96% correct cases. Furthermore is the logarithmic loss lesser for the decision tree classifier than for the SVM classifier.
This means the decision tree worked better than the SVM classifier for this data set.
