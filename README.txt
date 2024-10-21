Jayam Sutariya
CS 422

Neural Network

Implementations:

def softmax(z):
Helper function to calculate the softmax of z

def convert2D(arr):
Helper function that takes in the true-y label array and converts it to 2D

def calculate_loss(model, X, y):
Takes in the model, the data values, and the data labels, and calculates the loss based on current model
values. Function is fairly straightforward and based on the equation given in the project document.

def predict(model, x):
Takes in the model and x which can be one or many samples and predicts the output of the neural network.
This function is also fairly straighforward. Uses the hints given in the project document and uses the 
softmax helper function as well.

def build_model(X, y, nn_hdim, num_passes=20000, print_loss=True):
Takes in the data, the the data labels, number of nodes in the hidden layer, the number of epochs, and 
a flag that helps user decide whether or not to print the loss. Model values are randomly generted.
Uses back propagation (chain rule and gradient descent). Calculates certain variables needed for the chain
rule. Performs chain rule to get new model values. Performs gradient descent with the learning rate of 0.01.
Returns the model. Based on the print_loss, can output the Loss every 1000 epochs.