import os
import cv2 as cv
import numpy as np
import pickle
from PA4_utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from IPython.core.debugger import set_trace
import statistics
from statistics import mode

def build_vocabulary(image_paths, vocab_size = 100):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
  -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """
  # Load images from the training set. To save computation time, you don't
  # necessarily need to sample from all images, although it would be better
  # to do so. You can randomly sample the descriptors from each image to save
  # memory and speed up the clustering. Or you can simply call vl_dsift with
  # a large step size here, but a smaller step size in get_bags_of_sifts.
  #
  # For each loaded image, get some SIFT features. You don't have to get as
  # many SIFT features as you will in get_bags_of_sift, because you're only
  # trying to get a representative sample here.
  #
  # Once you have tens of thousands of SIFT features from many training
  # images, cluster them with kmeans. The resulting centroids are now your
  # visual word vocabulary.

  dim = 128      # length of the SIFT descriptors that you are going to compute.
  vocab = np.zeros((vocab_size,dim))
  
  for i in image_paths:
      
      # loading the image into gray-scale
      image = load_image_gray(i)
      
      # Computing the SIFT vectors of the feature points
      locations, descriptors = vlfeat.sift.dsift(image, step = 2, size = 8, fast = True)
      
      # limiting the number of key-points  by taking 50% of the detected key-points
      # In such a way larger images will have more key-points 
      descriptors = descriptors[:len(descriptors)//2]
      
      try :
          updated_descriptor = np.concatenate((updated_descriptor, descriptors))
      except NameError :
          updated_descriptor = descriptors
  
  print("SIFT descriptors computed")    
  
  # Computing the K-means clustering
  updated_descriptor = updated_descriptor.astype(float)
  vocab = vlfeat.kmeans.kmeans(updated_descriptor, vocab_size)
  
  print("Cluster centers computed")
  
  return vocab


def get_bags_of_sifts(image_paths, vocab_filename):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)
  
  IDF_list = np.zeros(vocab.shape[0])
  DF_list = np.zeros(vocab.shape[0])
    
    
  for i in image_paths:
      
      # dummy features variable
      feats = np.zeros(vocab.shape[0])
      
      # loading the image into gray-scale
      image = load_image_gray(i)
      
      # Computing the SIFT vectors of the feature points
      locations, descriptors = vlfeat.sift.dsift(image, step = 2, size = 8, fast = True)
      
      # limiting the number of key-points by taking 75% of the detected key-points
      # In such a way larger images will have more key-points 
      descriptors = descriptors[:len(descriptors)*3//4]
      
      # assigning the image features to corresponding cluster centers
      descriptors = descriptors.astype(float)
      assignments = vlfeat.kmeans.kmeans_quantize(descriptors, vocab)
      
      # creating the histogram
      for j in assignments:
          feats[j] += 1
          
      try :
          updated_image_feats = np.vstack((updated_image_feats, feats))
      except NameError :
          updated_image_feats = feats
  
  # TF-IDF implementation
  for i,j in enumerate(updated_image_feats):
      for k,l in enumerate(j):
          if l>0:
              DF_list[k] += 1
  
  for idx in range(0, len(IDF_list)):
          IDF_list[idx] = np.log(len(image_paths)/DF_list[idx])
  
  # incorporating IDF in histogram contruction
  tfidf_image_feats = updated_image_feats*IDF_list
  
  # Normalizing the histogram using L1 (Manhattan norm) 
  for i,j in enumerate(tfidf_image_feats):
      tfidf_image_feats[i] = j/np.linalg.norm(j, ord=1)

  return tfidf_image_feats
  '''  
  return updated_image_feats
  '''
def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean', K = 1):
  """
  This function will predict the category for every test image by finding
  the training image with most similar features. Instead of 1 nearest
  neighbor, you can vote based on k nearest neighbors which will increase
  performance (although you need to pick a reasonable value for k).

  Useful functions:
  -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
          -  X is a N x d numpy array of d-dimensional features arranged along
          N rows
          -  Y is a M x d numpy array of d-dimensional features arranged along
          N rows
          -  D is a N x M numpy array where d(i, j) is the distance between row
          i of X and row j of Y

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
          well for histograms
  -   K: The number of neighbours to be compared. Default is 1
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  test_labels = []

  # constructing the distance matrix
  D = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats, metric = metric)
  
  if K != 1:
      print("KNN neighbourhood size chosen : ", K)
      for i in range(0, D.shape[0]):
          category_list = []
          min_indices = sorted(range(len(D[i])), key = lambda sub: D[i][sub])[:K] 
          for i in min_indices:
              category_list.append(train_labels[i])
          try:
              mode_category = mode(category_list)
          except statistics.StatisticsError:
              mode_category = category_list[0]
          test_labels.append(mode_category)
  else :
      for i in range(0, D.shape[0]):
          test_labels.append(train_labels[np.argmin(D[i])])
      
  return  test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats):
  """
  This function will train a linear SVM for every category (i.e. one vs all)
  and then use the learned linear classifiers to predict the category of
  every test image. Every test feature will be evaluated with all 15 SVMs
  and the most confident SVM will "win". Confidence, or distance from the
  margin, is W*X + B where '*' is the inner product or dot product and W and
  B are the learned hyperplane parameters.

  Useful functions:
  -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  -   svm.fit(X, y)
  -   set(l)

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  
#  categories = list(set(train_labels))


#  svms = {cat: LinearSVC(random_state=0, tol=0.0001, loss='hinge', C=5, max_iter=5000) for cat in categories}
  
  svm = LinearSVC(random_state=0, tol=0.1, loss='hinge', C=5, max_iter=4000)
  
  # construct 1 vs all SVMs for each category ...
  # The Sklearn library has inbuilt implementation of OneVsRest Multiclass SVM classifier
  # Using the inbulit support as follows :
  test_labels = OneVsRestClassifier(svm).fit(train_image_feats, train_labels).predict(test_image_feats)
      

  return list(test_labels)
