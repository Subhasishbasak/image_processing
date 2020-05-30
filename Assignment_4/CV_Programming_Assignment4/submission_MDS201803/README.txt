All the implementations are done in the file named Bag_of_Features_code.py

It contains the following functions which has the following functionalities :

- build_vocabulary : Building the dictionary of visual words
- get_bag_of_sifts : Building the images features (feature histograms) for all training images
		   : Also implements the TF_IDF part for extra credit
- nearest_neighbor_classify : Implements image recognition with KNN algorithm
- svm_classify : Implements image recognition with SVM algorithm

The accuracy computation and constructing the confusion matrix is implemented in PA4_utils.py

It contains the following functions which has the following functionalities :

- show_results : computes accuracy and builds the confusion matrix

The code by default runs with the TF-IDF implementation. If one needs to run it without that, the TF_IDF part in "get_bag_of_sifts" needs to be commented out.

To run the codes just run all the cells in the notebook ProgAssignment4.ipynb
