*Project 1 - Supervised Learning
Algorithms to be implemented and learned: Decision Tree and Support Vector Machine (SVM)
Datasets: MNIST, CIFAR-10

Implement Decision Tree (you should manually set some hyperparameters involved).
- You may use DecisionTreeClassifier in the library or you can write your own one.
 - Each pixel is regarded as an attribute to construct a node.
 - Tree depth: 3, 6, 9, 12 (show all results with four different depths)
 - Use GridSearchCV to search proper hyperparameters (set cross validation k to 5).
 - Define the search pool for GridSearchCV as follows. (Note that the definition of the search pool can vary depending on the user's preferences or requirements.)
- Learn decision tree on training examples and run it for test examples.
- Show accuracy on training and test sets in a table.

*Project 2 - Unsupervised Learning
Algorithms to be implemented: Dimension reduction(PCA), Clustering(K-means, DBSCAN)
Datasets: Fashion MNIST

Implement k-means (you should manually set hyperparameters involved). 
- You may use KMeans in the library or you can write your own code from scratch. 
- Perform PCA to reduce the dimension before running the k-means algorithm (set dimension to 100, 50, and 10, respectively). Also run k-means for the original images (dimension of 784). 
- Set k (the number of clusters) to 10 (total number of classes in the dataset). 
- After running k-means with reduced dimensions, plot the 100 random chosen samples from the dataset using t-SNE (example below) – see above how to use t-SNE.  
     - If you specify the "hue" parameter as "Label" when using the "Implot" function, the function will automatically assign a color to each point based on its corresponding label.  
     - Show visualization results after k-means with four dimensions (784, 100, 50, 10).  
- Compute the clustering results using ARI (Adjusted Rand Index) for the four different dimensions (784, 100, 50, 10). Show results on test set (not train set) in a table. The ARI function is calculated by utilizing the actual labels and clustering labels using Sklearn's metrics package.  


Implement DBSCAN in a similar way to k-means. 
- You may use DBSCAN in the library or you can write your own code from scratch. 
- Set the hyperparameters (eps, minNeighbors). 
- Perform DBSCAN for four different dimensions (784, 100, 50, 10) and show clustering results on test set in a table and visualization results. 


*Project 3 - Deep Learning
Implementation codes in Implementation codes including model weights written in Python (in a zip file) for
two tasks, supervised learning and semi-supervised learning.

Task 1: Supervised Learning
- Image dataset of 96×96 size (number of classes: 50).
- We provide 30,000 images (each class has 600 images) as a labeled training set
and 2500 for a validation set.
- We do not provide the test set, which will be used to evaluate student’s
submissions by TA.


Task 2: Semi-Supervised Learning
- Image dataset of 96×96 size (number of classes: 10).
- We provide 6,000 images (each class has 600 images) as labeled training datasets,
6,000 images as unlabeled training sets, and 500 for validation sets.
