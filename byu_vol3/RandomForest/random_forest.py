"""
Random Forest Lab

Name
Section
Date
"""
import graphviz
import os
from uuid import uuid4

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""
    
    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]
    
    def match(self,sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        raise NotImplementedError('Problem 1 Incomplete')
        
    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(self.value))
    
def partition(data,question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    raise NotImplementedError('Problem 1 Incomplete')
    
#Problem 2    
def gini(data):
    """Return the Gini impurity of given array of data.
    Parameters:
        data (ndarray): data to examine
    Returns:
        (float): Gini impurity of the data"""
    raise NotImplementedError('Problem 2 incomplete')

def info_gain(left,right,G):
    """Return the info gain of a partition of data.
    Parameters:
        left (ndarray): left split of data
        right (ndarray): right split of data
        G (float): Gini impurity of unsplit data
    Returns:
        (float): info gain of the data"""
    raise NotImplementedError('Problem 2 Incomplete')
    
# Problem 3, Problem 7
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 7
    Returns:
        (float): Best info gain
        (Question): Best question"""
    raise NotImplementedError('Problem 3 Incomplete')

# Problem 4
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        raise NotImplementedError('Problem 4 Incomplete')

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
    #def __init__(self, question, right_branch, left_branch):
        raise NotImplementedError('Problem 4 Incomplete')

## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)    
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree):
    """Draws a tree"""
    #Remove the files if they already exist
    for file in ['Digraph.gv','Digraph.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf

# Prolem 5
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    raise NotImplementedError('Problem 5 Incomplete')

# Problem 6
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    raise NotImplementedError('Problem 6 Incomplete')
    
def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    raise NotImplementedError('Problem 6 Incomplete')

# Problem 7
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    raise NotImplementedError('Problem 7 Incomplete')

def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    raise NotImplementedError('Problem 7 Incomplete')

# Problem 8
def prob8():
    """Use the file parkinsons.csv to analyze a 5 tree forest.
    
    Create a forest with 5 trees and train on 100 random samples from the dataset.
    Use 30 random samples to test using analyze_forest() and SkLearn's 
    RandomForestClassifier.
    
    Create a 5 tree forest using 80% of the dataset and analzye using 
    RandomForestClassifier.
    
    Return three tuples, one for each test.
    
    Each tuple should include the accuracy and time to run: (accuracy, running time) 
    """
    raise NotImplementedError('Problem 8 Incomplete')