# CS 584 ML Project Fall 2021
This is our Project for CS 584 ML course at IIT Fall 2021 <br/>
The project's goal is to perform MNIST handwritten digits classification using hybrid random forests containing both generative and discriminative nodes. <br/>
Group Members: Marine Elisabeth and Roman Didelet

## Files

- Test_main: File containing all test functions to visualize results and different parts of the algorithm. The HOG descriptors extraction function was adapted from:<br/> https://github.com/ashar-7/hog_svm_mnist/blob/master/hog_svm_mnist.py
- ### SRC
  - NCMForest: File used to build the algorithm's overall Forest structure
  - NCMTree: File used to build the algorithm's tree structure and fill it with Node objects
  - Node: Files used to build Node objects and assign a classifier to each node.

- ### Headers
  - OneCentroid: Generative algorithm
  - NCMClassifier: Discriminative algorithm
  - utils: contains function to boostrap and bag samples for random forests

All commits reflect authorship except for three files: Test_main was done by both group members. <br/> NCMClassifier and utils were recovered from a previous project Roman worked on before IIT. Both files have mentions of their original author(s) at the top.
