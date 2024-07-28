# KMeans
This project demonstrates the comparison between Kmeans from scratch and from scikit-learn, and implements of applying two cluster algorithms to two different types of problems and data.

## cluster.py
A parent class for KMeans.py, including init and fit functions.

## KMeans.py
The implementation of KMeans from scratch. Functions includes:  
* init: the init function for the class  
* fit: the main function of the class  
* normalize: normalize the given data  
* cluster: the main clustering implementation  
* converge: check if the clustering converges or not  

## clustering.ipynb  
* The comparison of my own version of KMeans between the one from scikit-learn, including the figures and the v-measure score respectively  
* Kmeans on the Chicago taxi dataset with EDA, data preprocessing, data normalizing, hyperparameter tuning, clustering, and performance evaluation.
* DBSCAN on the routes of Mopsi users in Finland dataset with EDA, data preprocessing, data normalizing, hyperparameter tuning, clustering, and performance evaluation.  
