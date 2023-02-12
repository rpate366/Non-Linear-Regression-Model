import matplotlib.pyplot as plt
import numpy as np


def onesList(x):
    return [1] * x

def getError(x, y, w):
    sumOfTraining = np.sum(np.square((x@w) - y))
    return (sumOfTraining) / (len(x))

#Getting data from files
xTraining = open("hw1xtr.dat").read().splitlines()
xTraining = list(map(float, xTraining))
xTrainingOnes = np.matrix([onesList(len(xTraining)), xTraining, np.square(xTraining), np.power(xTraining, 3), np.power(xTraining, 4)]).getT()

yTraining = open("hw1ytr.dat").read().splitlines()
yTraining = list(map(float, yTraining))
yTraining = np.matrix(yTraining).getT()

xTest = open("hw1xte.dat").read().splitlines()
xTest = list(map(float, xTest))
xTestOnes = np.matrix([onesList(len(xTest)), xTest, np.square(xTest), np.power(xTest, 3), np.power(xTest, 4)]).getT()

yTest = open("hw1yte.dat").read().splitlines()
yTest = list(map(float, yTest))
yTest = np.matrix(yTest).getT()

XTXinv = np.linalg.inv(np.matmul(xTrainingOnes.getT(), xTrainingOnes))
XTy = np.matmul(xTrainingOnes.getT(), yTraining)
weight = np.matmul(XTXinv, XTy)

x = np.arange(xTrainingOnes.min(), xTrainingOnes.max() + 1, step = 0.1)
xN = np.matrix([onesList(len(x)) , x, np.square(x), np.power(x, 3), np.power(x, 4)]).getT()
line = (xN@weight).getT().A

fig = plt.figure()

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.title.set_text('Training')
ax2.title.set_text('Test')
ax1.set_xlabel("Error: " + str(getError(xTrainingOnes, yTraining, weight)))
ax2.set_xlabel("Error: " + str(getError(xTestOnes, yTest, weight)))

plt.subplot(121)
plt.plot(xTraining, yTraining.getT().A[0], 'ro')
plt.plot(x, line[0])
plt.ylim(0,25)
plt.xlim(-0.5,5)

plt.subplot(122)
plt.plot(xTest, yTest.getT().A[0], 'ro')
plt.plot(x, line[0])
plt.ylim(0,25)
plt.xlim(-0.5,5)

plt.show()