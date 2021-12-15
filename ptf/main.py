import numpy as np
import matplotlib.pyplot as plt

#number of iterations to run the algorithm
iterations = 20000

#used to clear the console to make things easier to read
def clearConsole(nbLines):
    for x in range(nbLines):
        print("")

 #simple sigmoid function
def sigmoid(x):
    return 1 / (1+ np.exp(-x))

#sigmoid derivative for backpropagation
def sigDeriv(x):
    return x * (1 - x )

def checkError(data, expected):
    print("")
    print("Testing ...")
    for x in range(np.size(data)):
        #print(round(float(data[x])))
        if float(data[x]) == 0.5:
            print("[~] L"+str(x)+" result:"+str(float(data[x]))+" expected:"+str(int(expected[x])))
        elif round(float(data[x])) == expected[x]:
            print("[O] L"+str(x)+" result:"+str(round(float(data[x])))+" expected:"+str(int(expected[x])))
        else:
            print("[X] L"+str(x)+" result:"+str(round(float(data[x])))+" expected:"+str(int(expected[x])))







#inputs
inputs = np.array([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1],
    ])
#expected outputs
expectedOutputs = np.array([[0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]]).T
#seed to compare results
np.random.seed(5)

#random weight between -1 and 1 
weights = 2 * np.random.rand(4,1) - 1
clearConsole(20)
print("-------------------------------------------")
print("Running " + str(iterations) + " iterations:")
print("Inputs")
print(inputs)

print(" ")
print("Random weights: ")
print(weights)

for x in range(iterations):
    inputsLayer = inputs

    outputs = sigmoid(np.dot(inputsLayer, weights))

    error = outputs - expectedOutputs

    corrective = error * sigDeriv(outputs)

    weights += np.dot(inputsLayer.T, corrective)

checkError(outputs, expectedOutputs)

print("")
print("Weights after training")
print(weights)

print(" ")
print("Outputs")
print(outputs)




"""#inputs
inputs = np.array([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1],
    ])
#expected outputs
expectedOutputs = np.array([[0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1]]).T
"""