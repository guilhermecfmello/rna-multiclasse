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
        outputs = np.zeros(shape=(nRows, 5), dtype=np.double)
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
  
  def testDatabase(self, dbName):
    try:
      with open(dbName) as db:
        csvInput = csv.reader(db, delimiter=',', quotechar='|')
        nRows = sum(1 for line in csvInput)
        db.seek(0)
        inputs = np.ones(shape=(nRows, 2), dtype=np.double)
        outputs = np.zeros(shape=(nRows, 5), dtype=np.double)
        i = 0
        for c in csvInput:
          inputs[i][0] = c[0]
          inputs[i][1] = c[1]
          outputs[i][0] = c[2]
          outputs[i][1] = c[3]
          outputs[i][2] = c[4]
          outputs[i][3] = c[5]
          outputs[i][4] = c[6]
          i = i + 1
        self.inputsTest = inputs
        self.outputsTest = outputs
        self.nTest = len(inputs)
    except IOError:
      print("Error openning database file: " + str(IOError))

  def trainning(self, ages = 0, lr = 0.1):
    self.__setWeights()
    for a in range(ages):
      for i in range(self.n):
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
    self.weights = rd.uniform(-1,1,[5,2])
    self.bias = rd.uniform(-1,1,[5])
  
  # sigmoid
  def __sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def compare(self):
    # positives rights
    pr = 0 
    # positives wrongs
    pw = 0
    # negatives rights
    nr = 0
    # negatives wrong
    nw = 0
    for i in range(self.n):
      y = self.__sigmoid(np.dot(self.weights, self.inputs[i])+self.bias)
      for j in range(len(y)):
        y[j] = 1. if y[j] > 0.5 else 0.
      if(self.outputs[i][0] == 1.):
        if y[0] == self.outputs[i][0]:
          pr = pr + 1
        else:
          pw = pw + 1
      elif self.outputs[i][1] == 1.:
        if y[1] == self.outputs[i][1]:
          nr = nr + 1
        else:
          nw = nw + 1
      elif self.outputs[i][2] == 1.:
        if y[2] == self.outputs[i][2]:
          nr = nr + 1
        else:
          nw = nw + 1
      elif self.outputs[i][3] == 1.:
        if y[3] == self.outputs[i][3]:
          nr = nr + 1
        else:
          nw = nw + 1
      elif self.outputs[i][4] == 1.:
        if y[4] == self.outputs[i][4]:
          nr = nr + 1
        else:
          nw = nw + 1

    # database positives
    pN = np.zeros([5])
    # database negatives
    nN = np.zeros([5])
    for i in range(len(self.outputs)):
      if self.outputs[i][0] == 1:
        pN[0] = pN[0] + 1
      elif self.outputs[i][1] == 1:
        pN[1] = pN[1] + 1
      elif self.outputs[i][2] == 1:
        pN[2] = pN[2] + 1
      if self.outputs[i][0] == 0:
        nN[0] = nN[0] + 1
      elif self.outputs[i][1] == 0:
        nN[1] = nN[1] + 1
      elif self.outputs[i][2] == 0:
        nN[2] = nN[2] + 1
      # nN = nN + 1
    rights = pr+nr
    wrongs = pw+nw
    precision = pr/(pr+pw)
    revocation = pr/(pr+pw)
    accuracy = (pr+nr)/(pr+pw+nr+nw)
    # F1 = 2*(precision*revocation)/(precision+revocation)
    print(" ===== Neural Network =====")
    # print("    w1: " + str(self.weights[0]) + "  w2: " + str(self.weights[1]))
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
    # print("F1: " + str(F1))
    return rights, wrongs

  def compareTest(self):
    # positives rights
    pr = 0
    # positives wrongs
    pw = 0
    # negatives rights
    nr = 0
    # negatives wrong
    nw = 0
    for i in range(len(self.inputsTest)):
      y = self.__sigmoid(np.dot(self.weights, self.inputsTest[i])+self.bias)
      for j in range(len(y)):
        y[j] = 1. if y[j] > 0.5 else 0.
      if(self.outputsTest[i][0] == 1.):
        if y[0] == 1:
          pr = pr + 1
        else:
          pw = pw + 1
      elif self.outputsTest[i][1] == 1.:
        if y[1] == 1:
          pr = pr + 1
        else:
          pw = pw + 1
      elif self.outputsTest[i][2] == 1.:
        if y[2] == 1:
          pr = pr + 1
        else:
          pw = pw + 1

    # database positives
    pN = np.zeros([3])
    # database negatives
    nN = np.zeros([3])
    for i in range(len(self.outputsTest)):
      if self.outputsTest[i][0] == 1:
        pN[0] = pN[0] + 1
      elif self.outputsTest[i][1] == 1:
        pN[1] = pN[1] + 1
      elif self.outputsTest[i][2] == 1:
        pN[2] = pN[2] + 1
      if self.outputsTest[i][0] == 0:
        nN[0] = nN[0] + 1
      elif self.outputsTest[i][1] == 0:
        nN[1] = nN[1] + 1
      elif self.outputsTest[i][2] == 0:
        nN[2] = nN[2] + 1
      # nN = nN + 1
    # print(pr)
    # print(pw)
    # print(nr)
    # print(nw)
    # print('range: ' + str(len(self.outputsTest[0])))
    # for j in range(len(self.outputsTest[0])):
    rights = pr+nr
    wrongs = pw+nw
    print('pr: ' + str(pr))
    print('pw: ' + str(pw))
    # print(type(pr))
    # print(type(pw))
    precision = pr/(pr+pw)
    revocation = pr/(pr+pw)
    if j == 1: exit()
    accuracy = (pr+nr)/(pr+pw+nr+nw)
    # F1 = 2*(precision*revocation)/(precision+revocation)
    print("\n\n\n\n++++++++++++++++++\n\n\n\n")
    print("===== Neural Network =====")
    # print("    w1: " + str(self.weights[0]) + "  w2: " + str(self.weights[1]))
    print("")
    print("Total: " + str(rights + wrongs))
    print("Rights: " + str(rights) + " Wrongs: " + str(wrongs))
    # print("Percent rights: " + str((rights/len(inputsTestTest))*100) + "%")
    print("Positives rights: " + str(pr))
    print("Positives wrongs: " + str(pw))
    print("Negatives rights: " + str(nr))
    print("Negatives wrongs: " + str(nw))
    print("Precision: " + str(precision))
    print("Revocation: " + str(revocation))
    print("Accuracy: " + str(accuracy))
    # print("F1: " + str(F1))
    return rights, wrongs


  def plot_all(self):
    # print(self.outputs[:,0])
    # print("++++++++++++++++++++++++")
    # print(self.inputs)
    # print("++++++++++++++++++++++++")
    # print(class1_X)
    # exit()
    class1_X = np.take(self.inputs[:,0], np.where(self.outputs[:,0] == 1))
    class1_Y = np.take(self.inputs[:,1], np.where(self.outputs[:,0] == 1))
    class2_X = np.take(self.inputs[:,0], np.where(self.outputs[:,1] == 1))
    class2_Y = np.take(self.inputs[:,1], np.where(self.outputs[:,1] == 1))
    class3_X = np.take(self.inputs[:,0], np.where(self.outputs[:,2] == 1))
    class3_Y = np.take(self.inputs[:,1], np.where(self.outputs[:,2] == 1))
    plt.plot(class1_X, class1_Y, "+r")
    plt.plot(class2_X, class2_Y, "+b")
    plt.plot(class3_X, class3_Y, "+y")
    xx = np.linspace(-3, 4)
    for l in range(3):
      for k in np.linspace(np.amin(self.inputs[:,:1]),np.amax(self.inputs[:,:1])):
        slope = -(self.bias[l]/self.weights[l][1])/(self.bias[l]/self.weights[l][1])  
        intercept = -self.bias[l]/self.weights[l][1]
        # y =mx+c, m is slope and c is intercept
        yLinha = (slope*k) + intercept
        plt.plot(k, yLinha,'ko')
    # print(self.weights)
    # print(self.weights[0][1])
    # exit()
    plt.plot(xx, (self.weights[0][0] * xx + self.weights[0][1] + self.bias[0]), "red")
    plt.plot(xx, (self.weights[1][0] * xx + self.weights[1][1] + self.bias[1]), "blue")
    plt.plot(xx, (self.weights[2][0] * xx + self.weights[2][1] + self.bias[2]), "yellow")

    plt.show()

