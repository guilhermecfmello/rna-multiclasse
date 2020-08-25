import csv
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

class Neural:
  # Constructor class must receive a valid database name
  def __init__(self, dbName):
    try:
      with open(dbName) as db:
        csvInput = csv.reader(db, delimiter=',', quotechar='|')
        nRows = sum(1 for line in csvInput)
        db.seek(0)
        inputs = np.ones(shape=(nRows, 2), dtype=np.double)
        outputs = np.zeros(shape=(nRows, 3), dtype=np.double)
        i = 0
        for c in csvInput:
          inputs[i][0] = c[0]
          inputs[i][1] = c[1]
          outputs[i][0] = c[2]
          outputs[i][1] = c[3]
          outputs[i][2] = c[4]
          i = i + 1
        self.inputs = inputs
        self.outputs = outputs
        self.n = len(inputs)
    except IOError:
      print("Error openning database file: " + str(IOError))
  
  def trainning(self, ages = 0, lr = 0.1):
    self.__setWeights()
    for a in range(ages):
      for i in range(self.n):
        # calculating sum with bias
        yCalc = self.__sigmoid(np.dot(self.weights, self.inputs[i])+self.bias)
        error = self.__calcQuadracticError(yCalc, self.outputs[i], self.inputs[i])
        errorBias = self.__calcQuadracticErrorBias(yCalc, self.outputs[i])
        self.weights = self.weights - lr/self.n * error
        self.bias = self.bias - lr/self.n * errorBias
    return True

  def __calcQuadracticErrorBias(self, yCalc, y):
    a = 2*yCalc - 2*y
    b = yCalc*(1-yCalc)
    c = (a*b)
    return c
  
  def __calcQuadracticError(self, yCalc, y, x):
    a = 2*yCalc - 2*y
    b = yCalc*(1-yCalc)
    c = (a*b).reshape((1,-1))
    result = np.matmul(x[None].T, c)
    return result.T
    
  def __derivative(self, y, d):
    return (((2*y) - (2*d)) * (y*(1-y)))
    
  # Define random weights with bias
  def __setWeights(self):
    self.weights = rd.uniform(-1,1,[3,2])
    self.bias = rd.uniform(-1,1,[3])
  
  # sigmoid
  def __sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def compare(self, weights, bias):
    # positives rights
    pr = 0 
    # positives wrongs
    pw = 0
    # negatives rights
    nr = 0
    # negatives wrong
    nw = 0
    for i in range(len(self.outputs)):
        y = sigmoid(linear_combination(inputs[i],weights, bias))
        y = 1. if y > 0.5 else 0.
        if(self.outputs[i] == 1.):
            if y == outputs[i]:
                pr = pr + 1
            else:
                pw = pw + 1
        else:
            if y == outputs[i]:
                nr = nr + 1
            else:
                nw = nw + 1
    # database positives
    pN = 0
    # database negatives
    nN = 0
    for i in range(len(outputs)):
        if outputs[i] == 1:
            pN = pN + 1
        else:
            nN = nN + 1
    rights = pr+nr
    wrongs = pw+nw
    precision = pr/(pr+pw)
    revocation = pr/(pr+pw)
    accuracy = (pr+nr)/(pr+pw+nr+nw)
    F1 = 2*(precision*revocation)/(precision+revocation)
    print(" ===== Neural Network =====")
    print("    w1: " + str(weights[0]) + "  w2: " + str(weights[1]))
    print("")
    print("Total: " + str(rights + wrongs))
    print("Rights: " + str(rights) + " Wrongs: " + str(wrongs))
    # print("Percent rights: " + str((rights/len(inputs))*100) + "%")
    print("Positives rights: " + str(pr))
    print("Positives wrongs: " + str(pw))
    print("Negatives rights: " + str(nr))
    print("Negatives wrongs: " + str(nw))
    print("============================")
    print("Precision: " + str(precision))
    print("Revocation: " + str(revocation))
    print("Accuracy: " + str(accuracy))
    print("F1: " + str(F1))
    return rights, wrongs

  def plot_all(self, X, Y, w):
    pos_X = np.take(X[:, 0], np.where(Y == 1))
    pos_Y = np.take(X[:, 1], np.where(Y == 1))
    neg_X = np.take(X[:, 0], np.where(Y == 0))
    neg_Y = np.take(X[:, 1], np.where(Y == 0))
    plt.plot(pos_X, pos_Y, "+r")
    plt.plot(neg_X, neg_Y, "+b")
    xx = np.linspace(-3, 4)  
    plt.plot(xx, (w[0] * xx + w[1]), "green")  
    plt.show()

  