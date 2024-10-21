import numpy as np

#Softmax helper function
def softmax(z):
    numerator = np.exp(z - np.max(z))
    denominator = np.sum(numerator)
    return numerator/denominator

#Convert True y array into 2d array
def convert2D(arr):
    y2d = np.zeros((len(arr), 2))
    for c in range(len(arr)):
        if arr[c] == 0:
            y2d[c][0] = 1
        else:
            y2d[c][1] = 1
    
    return y2d

#Calculate Loss
def calculate_loss(model, X, y):
    yNew = convert2D(y)

    predicted = []
    for i in range(len(X)):
        predicted.append(predict2(model, X[i]))

    yHat = np.array(predicted)
    loss = (1/len(X)) * (-np.sum(yNew * np.log(yHat)))

    return loss

#Predict function used in test file
def predict(model, x):
    alpha = np.matmul(x, model['W1']) + model['b1']
    h = np.tanh(alpha)
    zeta = np.matmul(h, model['W2']) + model['b2']
    yhat = softmax(zeta)
    
    return np.argmax(yhat, axis=1)

#Predict function used for the rest of this file
def predict2(model, x):
    alpha = np.matmul(x, model['W1']) + model['b1']
    h = np.tanh(alpha)
    zeta = np.matmul(h, model['W2']) + model['b2']
    yhat = softmax(zeta)
    
    return yhat

#Build the model
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=True):
    W1 = np.random.rand(2, nn_hdim)
    W2 = np.random.rand(nn_hdim, 2)
    b1 = np.random.rand(nn_hdim)
    b2 = np.random.rand(2)

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    yNew = convert2D(y)

    for i in range(num_passes):
        #Prior calculations
        yHat = predict2(model, X)

        alpha = np.matmul(X, model['W1']) + model['b1']
        h = np.tanh(alpha)

        #Partial Derivative calculations
        L_yHat = np.subtract(yHat, yNew)
        L_a = np.multiply( ( 1 - pow(np.tanh(alpha), 2) ) , ( np.matmul(L_yHat, W2.T) ) )

        partW2 = np.matmul(np.transpose(h), L_yHat)
        partb2 = np.sum(L_yHat)
        partW1 = np.matmul(np.transpose(X), L_a)
        partb1 = np.sum(L_a)

        #Updates
        W1 = np.subtract(W1, np.multiply(0.01, partW1))
        W2 = np.subtract(W2, np.multiply(0.01, partW2))
        b1 = np.subtract(b1, np.multiply(0.01, partb1))
        b2 = np.subtract(b2, np.multiply(0.01, partb2))

        model['W1'] = W1
        model['W2'] = W2
        model['b1'] = b1
        model['b2'] = b2

        #Loss output
        if print_loss == True:
            if ( ((i % 1000) == 0) & (i != 0) ):
                print("Loss after", i, "iterations:", calculate_loss(model, X, y))
            if (i == (num_passes - 1)):
                print("Loss after", i+1, "iterations:", calculate_loss(model, X, y))

    return model