import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


class NMFRecommender:

    def __init__(self,):
        """The parameter values for the algorithm"""
  
       
    def initialize_matrices(self):
        """Initialize the W and H matrices"""

        
    def compute_loss(self):
        """Computes the loss of the algorithm according to the frobenius norm"""

    
    def update_matrices(self):
        """The multiplicative update step to update W and H"""

      
    def fit(self):
        """Fits W and H weight matrices according to the multiplicative update 
        algorithm. Return W and H"""
        

    def reconstruct(self):
        """Reconstructs the V matrix for comparison against the original V 
        matrix"""


        
def prob4():
    """Run NMF recommender on the grocery store example"""
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])


def prob5():
    """Calculate the rank and run NMF
    """

def discover_weekly():
   """
  Create the recommended weekly 30 list for a given user
  """