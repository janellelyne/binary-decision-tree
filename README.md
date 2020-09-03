Project Scope

This project is a Python implementation of a binary decision tree from scratch.

The ID3 algorithm to build a decision tree mainly consists of using a calculated hueristic to split the data at each node in the tree. For this project two hueristics were used:
  1. Information Gain 
     Information gain is the main key that is used by Decision Tree Algorithms to construct a Decision Tree. Decision Trees algorithm will      always tries to maximize Information gain. An attribute with highest Information gain will tested/split first(become the root).
  2. Variance Impurity
     Let K denote the number of examples in the training set. Let K0 denote the number of training examples that have class = 0 and K1          denote the number of training examples that have class = 1.
     The variance impurity of the training set S is defined as:
        𝑉𝐼(𝑆) = 𝐾0/K * K1/K

This project builds two trees using the huerstics mentioned above, runs tests against the trees to produce the accurac, and implements post-pruning to increase the accuracy.  

Summary of Tasks:
- Implement the decision tree learning algorithm. As discussed in class, the main
step in decision tree learning is choosing the next attribute to split on. 
- Implement the following two heuristics for selecting the next attribute.
  1. Information gain heuristic.
  2. Variance impurity heuristic.
- Output the accuracies on the test set for decision trees constructed using the
two heuristics as well as the accuracies for their post-pruned versions for the given
values of L and K. If to-print equals yes, it should print the decision tree in the format
described above to the standard output.
- Report the accuracy on the test set for decision trees constructed using the two
heuristics mentioned above.
- Choose 10 suitable values for L and K (not 10 values for each, just 10
combinations). For each of them, report the accuracies for the post-pruned decision
trees constructed using the two heuristics.

