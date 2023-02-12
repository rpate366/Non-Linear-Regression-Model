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


regParam = [0.01, 0.05, 0.1, 0.5, 1, 100, 10**6]
l = []
for i in range(len(xTrainingOnes.A[0])):
    temp = []
    for j in range(len(xTrainingOnes.A[0])):
        temp.append(0)
    l.append(temp)
    if i != 0:
        l[i][i] = 1

errorsTraining = []
errorsTest = []
weight = []
for i in range(len(regParam)):
    XTX = np.matmul(xTrainingOnes.getT(), xTrainingOnes)
    regParamI = np.multiply(regParam[i], l)
    XTy = np.matmul(xTrainingOnes.getT(), yTraining)

    weight.append(np.matmul(np.linalg.inv(XTX + regParamI), XTy))
    errorsTraining.append(getError(xTrainingOnes, yTraining, weight[i]))
    errorsTest.append(getError(xTestOnes, yTest, weight[i]))

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax1.title.set_text('Training Error vs Lambda')
ax2.title.set_text('Test Error vs Lambda')
ax3.title.set_text('Weight vs Lambda')

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

plt.subplot(221)
plt.plot(regParam, errorsTraining, 'ro')
plt.xscale("log")

plt.subplot(222)
plt.plot(regParam, errorsTest, 'ro')
plt.xscale("log")

plt.subplot(223)
for x, y in zip(regParam, weight):
    plt.plot([x] * len(y), y, marker = ".")
plt.xscale("log")


plt.show()

#0.01 is best for training
#0.1 is best for test