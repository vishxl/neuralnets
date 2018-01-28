import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

class Config:
    nn_input_dim = 2  
    nn_output_dim = 2  
   
    epsilon = 0.01 
    reg_lambda = 0.01 
    
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y
