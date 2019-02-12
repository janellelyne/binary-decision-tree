import pandas as pd
import numpy as np
import random as random
import copy as copy
import sys

def get_attr_list(arg):
  attr = list(arg)
  train = arg[2:]
  target = attr[-1]
  del attr[-1]
  return train, attr, target
  
def get_best_attr_ig(data, attr, target):
  pos, neg, total = calc_impurity(-1, data)
  entropy_s = calc_entropy(pos, neg, total)
  bestind = best_info_gain = best_attr = 0 
  for index, attribute in enumerate(attr):
    info_gain = calc_info_gain(entropy_s, attribute, data, target)
    if info_gain > best_info_gain:
      best_info_gain = info_gain
      bestind = index
      best_attr = attribute
  return bestind

def get_best_attr_vi(data, attr, target):
  pos, neg, total = calc_impurity(-1, data)
  variance_s = calc_variance(pos, neg, total)
  bestind = best_info_gain = best_attr = 0 
  for index, attribute in enumerate(attr):
    info_gain = calc_info_gain_v(variance_s, attribute, data, target)
    if info_gain > best_info_gain:
      best_info_gain = info_gain
      bestind = index
      best_attr = attribute
  return bestind

def calc_impurity(index, data):
  neg = 0
  pos = 0
  total = 0
  pos = (data[data.columns[index]] == 1).sum()
  neg = (data[data.columns[index]] == 0).sum()
  total = data[data.columns[index]].count()
  return pos, neg, total

def calc_attr_impurity(val, index, data, target):
  neg = 0
  pos = 0
  total = 0
  if val == 1:
      pos = len(data[(data[index] == 1) & (data[target] == 1)])
      neg = len(data[(data[index] == 1) & (data[target] == 0)])
      total = len(data[(data[index] == 1)])
  else:
      pos = len(data[(data[index] == 0) & (data[target] == 1)])
      neg = len(data[(data[index] == 0) & (data[target] == 0)])
      total = len(data[(data[index] == 0)])
                       
  return pos, neg, total


def calc_entropy(pos, neg, total):
  if total == 0:
      return 0      
  pos_prob = pos / total
  neg_prob = neg / total
  if pos_prob == 0:
    return 0
  if neg_prob == 0:
    return 0
  return -pos_prob * np.log2(pos_prob) - neg_prob * np.log2(neg_prob)

def calc_variance(pos, neg, total):
  if total == 0:
      return 0
  pos_prob = pos / total
  neg_prob = neg / total
  if pos_prob == 0:
    return 0
  if neg_prob == 0:
    return 0
  return pos_prob * neg_prob
   

def calc_info_gain(entropy_s, index, data, target):
  pos_a, neg_a, total_a = calc_attr_impurity(1, index, data, target)
  pos_b, neg_b, total_b = calc_attr_impurity(0, index, data, target)
  
  total_s = len(data)
  entropy_a = calc_entropy(pos_a, neg_a, total_a)
  entropy_b = calc_entropy(pos_b, neg_b, total_b)
  weighted_average = total_a /total_s * entropy_a + total_b / total_s * entropy_b
  return entropy_s - weighted_average

def calc_info_gain_v(variance_s, index, data, target):
  pos_a, neg_a, total_a = calc_attr_impurity(1, index, data, target)
  pos_b, neg_b, total_b = calc_attr_impurity(0, index, data, target)
  
  total_s = len(data)
  variance_a = calc_variance(pos_a, neg_a, total_a)
  variance_b = calc_variance(pos_b, neg_b, total_b)
  weighted_average = total_a /total_s * variance_a + total_b / total_s * variance_b
  return variance_s - weighted_average
  
def partition(best, attr_list, training):
  pos = training.loc[training[attr_list[best]] == 1]
  neg = training.loc[training[attr_list[best]] == 0]
  return pos, neg

def build_tree(train, target, attr_list, best_type):
  root = Node()
  if len(train[train[target] == 1]) == len(train):
    root.set_name(1)
    return root
  if len(train[train[target] == 0]) == len(train):
    root.set_name(0)
    return root
  if not attr_list:
    pos = len(train[train[target] == 1])
    neg = len(train[train[target] == 0])
    if pos > neg:
      root.set_name(1)
      return root
    elif neg >= pos:
      root.set_name(0)
      return root
  if best_type == "ig":
      best = get_best_attr_ig(train, attr_list, target)
  else:
      best = get_best_attr_vi(train, attr_list, target)
  
  root = Node(attr_list[best]) 
  pos_train, neg_train = partition(best, attr_list, train)
  name_of_Attr = attr_list[best]
  root.set_nums_of_attr(len(pos_train), len(neg_train))
  
  if len(pos_train) == 0:
    pos = len(train[train[target] == 1])
    neg = len(train[train[target] == 0])
    if pos > neg:
      root.set_name(1)
      return root
    elif neg >= pos:
      root.set_name(0)
      return root
  else:
    attr_list_p = attr_list.copy()
    del attr_list_p[best]
    root.set_right(build_tree(pos_train, target, attr_list_p, best_type))
  if len(neg_train) == 0:
    pos = len(train[train[target] == 1])
    neg = len(train[train[target] == 0])
    if pos > neg:
      root.set_name(1)
      return root
    elif neg >= pos:
      root.set_name(0)
      return root
  else:
    attr_list_n = attr_list.copy()
    del attr_list_n[best]
    root.set_left(build_tree(neg_train, target, attr_list_n, best_type))
  return root


def print_tree(node, spacing="|"):
    if node == None:
      return print("")
    if node.get_right() == None and node.get_left() == None:
        return print(str(node.get_name()))
    print("")
    # Print the question at this node
    print (spacing + str(node.get_name()) + " = 0:", end ="")
    print_tree(node.get_left(), spacing + "|") 
   
    # Call this function recursively on the true branch
    print (spacing + str(node.get_name()) + " = 1:", end ="" )
    print_tree(node.get_right(), spacing + "|")

def print_tree_file(node, file, spacing = "|"):
    if node == None:
      return file.write("\n")
    if node.get_right() == None and node.get_left() == None:
        return file.write(str(node.get_name()) + "\n")
    file.write("\n")
    # Print the question at this node
    file.write(spacing + str(node.get_name()) + " = 0:")
    print_tree_file(node.get_left(), file, spacing + "|") 
   
    # Call this function recursively on the true branch
    file.write(spacing + str(node.get_name()) + " = 1:")
    print_tree_file(node.get_right(), file, spacing + "|")
    
    
def test_tree(tree, training):
  pos = 0
  for (index, row) in training.iterrows():
    predict = classify(tree, row)
    actual = row[-1]
    if predict ==actual:
      pos = pos + 1
  return pos / len(training)
    
    
def classify(root, row):
  if root == None:
    return
  if root.get_right() == None and root.get_left() == None:
        return root.get_name()
  if row[root.get_name()] == 0:
    return classify(root.get_left(), row)
  elif row[root.get_name()] == 1:
    return classify(root.get_right(), row)
  
def post_pruning(L, K, tree1, best_accuracy, valid):
  D_best = copy.deepcopy(tree1)
  for x in range(1, L):
    D_dash = copy.deepcopy(D_best)
    M = random.randint(1, K)
    for j in range(1, M):
      list_order = get_node_ordering(D_dash)
      num_nodes = len(list_order)
      P = random.randint(1, num_nodes) - 1
      pos, neg = list_order[P].get_nums_of_attr()
      if pos > neg:
        list_order[P].set_name(1)
      else:
        list_order[P].set_name(0)
    accuracy = test_tree(D_dash, valid)
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      D_best = copy.deepcopy(D_dash)
  return D_best, best_accuracy
    
def get_node_ordering(root):
  node_list = []
  if root:
     if root.get_left() != None and root.get_right() != None:
        node_list.append(root)
        node_list = node_list + get_node_ordering(root.get_left())
        node_list = node_list + get_node_ordering(root.get_right())
  return node_list
 
def generate_report(tree1, tree2, tree3, tree4, ig_accuracy, vi_accuracy, new_accuracy_ig, new_accuracy_vi, valid):
    f = open("Report.txt", "w")
    f.write("This is a report file for the ML Decision Tree assignment.\n\n")
    f.write("Tree #1: Information Gain Hueristic.\nThis tree had an initial"
            + " accuracy of: " + str(ig_accuracy) + "\nThe post-pruned tree had an"
            + " accuracy of: " + str(new_accuracy_ig) + "\n")
    f.write("\nThe resulting trees are listed below:\n")
    print_tree_file(tree1, f)
    print_tree_file(tree3, f)
    f.write("\nTree #2: Variance Impurity Hueristic.\n This tree had an initial"
            + " accuracy of: " + str(vi_accuracy) + "\nThe post-pruned tree had an"
            + " accuracy of: " + str(new_accuracy_vi))
    f.write("\nThe resulting trees are listed below: ")
    print_tree_file(tree2, f)
    print_tree_file(tree4, f)
    
    f.write("\n\nRandomly choosing ten combinations of values L and K in range 1 to 100"
            + ",\nthis is the resulting accuracies for post-pruning on the"
            + " tree with the \nInformation Gain Hueristic:")
    for x in range(1,10):
        L = random.randint(1, 100)
        K = random.randint(100, 100)
        new_tree, new_acc = post_pruning(L, K, tree1, ig_accuracy, valid)
        f.write("\n\nPost Pruning Iteration " + str(x) + ":\n"
                + "L: " + str(L) + " K: " + str(K)
                + "\nPrevious Accuracy: " + str(ig_accuracy)
                + "\nPost Pruned Accuracy: " + str(new_acc))
    f.write("\n\nRandomly choosing ten combinations of values L and K in range 1 to 100"
            + ",\nthis is the resulting accuracies for post-pruning on the"
            + " tree with the \nVariance Impurity Hueristic:")
    for x in range(1,11):
        L = random.randint(1, 100)
        K = random.randint(1, 100)
        new_tree, new_acc = post_pruning(L, K, tree2, vi_accuracy, valid)
        f.write("\n\nPost Pruning Iteration " + str(x) + ":\n"
                + "L: " + str(L) + " K: " + str(K)
                + "\nPrevious Accuracy: " + str(vi_accuracy)
                + "\nPost Pruned Accuracy: " + str(new_acc))
    f.close()
      
  
class Node:
  def __init__(self, name = None):
    self.right = None
    self.left = None
    self.name = name
    self.pos = 0
    self.neg = 0
  def set_right(self, arg):
    self.right = arg
  def set_left(self, arg):
    self.left = arg
  def set_name(self, arg):
    self.name = arg
  def get_right(self):
    return self.right
  def get_left(self):
    return self.left
  def get_name(self):
    return self.name
  def set_nums_of_attr(self, pos, neg):
    self.pos = pos
    self.neg = neg
  def get_nums_of_attr(self):
    return self.pos, self.neg

def main(L, K, train, valid, test, to_print):
  #This is my main function
  train = pd.read_csv(train)
  test = pd.read_csv(test)
  valid = pd.read_csv(valid)
  valid = valid[2:]
  training, attr_list, target = get_attr_list(train)
  print("Please wait while program runs... est. [25] second run time...\n\n")
  tree1 = build_tree(training, target, attr_list, "ig")
  tree2 = build_tree(training, target, attr_list, "vi")
  if to_print.upper() == "YES": 
      print("Tree #1 with Information Gain Hueristic: ")
      print_tree(tree1)
      print("Tree #2 with Variance Impurity Hueristic: ")
      print_tree(tree2)   
  test = test[2:]
  ig_accuracy = test_tree(tree1, test)
  vi_accuracy = test_tree(tree2, test)
  print("Prediction Accuracy with Information Gain Hueristic: " + str(ig_accuracy))
  print("Prediction Accuracy with Variance Impurity Hueristic: " + str(vi_accuracy))
  tree3, new_accuracy_ig = post_pruning(L, K, tree1, ig_accuracy, valid)
  tree4, new_accuracy_vi = post_pruning(L, K, tree2, vi_accuracy, valid)
  print("Post Pruned Prediction Accuracy for Information Gain Hueristic: " + str(new_accuracy_ig))
  print("Post Pruned Prediction Accuracy for Variance Impurity Hueristic: " + str(new_accuracy_vi))
  
  #Generate report file if neccessary
  #generate_report(tree1, tree2, tree3, tree4, ig_accuracy, vi_accuracy, new_accuracy_ig, new_accuracy_vi, valid)

'''
For use with notebooks:
main(10, 4, "training_set.csv", validation_set.csv", test_set.csv", "yes")
'''
if __name__== "__main__":
   main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])






