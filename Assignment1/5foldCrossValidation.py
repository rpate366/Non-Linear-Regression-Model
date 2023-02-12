import matplotlib.pyplot as plt
import numpy as np
import copy

def onesList(x):
    return [1] * x

def getError(x, y, w):
    sumOfTraining = np.sum(np.square((x@w) - y))
    return (sumOfTraining) / (len(x))

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax1.title.set_text('Error vs Lambda')
ax2.title.set_text('Test Data')

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

#Getting data from files
xTraining = open("hw1xtr.dat").read().splitlines()
xTraining = list(map(float, xTraining))

arraySets = np.array_split(xTraining, 5)
bestLambda = []

for block in range(len(arraySets) - 1):
    validationSet = arraySets[block]
    xTrainingTemp = copy.deepcopy(xTraining)
    
    for v in validationSet:
        xTrainingTemp.remove(v)

    xTrainingOnes = np.matrix([onesList(len(xTrainingTemp)), xTrainingTemp, np.square(xTrainingTemp), np.power(xTrainingTemp, 3), np.power(xTrainingTemp, 4)]).getT()

    yTraining = open("hw1ytr.dat").read().splitlines()
    yTraining = list(map(float, yTraining))
    ySplit = np.array_split(yTraining, 5)

    for v in ySplit[block]:
        yTraining.remove(v)
    yTraining = np.matrix(yTraining).getT()

    regParam = [0.01, 0.05, 0.1, 0.5, 1, 100, 10**6]
    I = []
    for i in range(len(xTrainingOnes.A[0])):
        temp = []
        for j in range(len(xTrainingOnes.A[0])):
            temp.append(0)
        I.append(temp)
        if i != 0:
            I[i][i] = 1

    errorsTraining = []
    weight = []
    validationSet = np.matrix([onesList(len(validationSet)), validationSet, np.square(validationSet), np.power(validationSet, 3), np.power(validationSet, 4)]).getT()

    for i in range(len(regParam)):
        XTX = np.matmul(xTrainingOnes.getT(), xTrainingOnes)
        regParamI = np.multiply(regParam[i], I)
        XTy = np.matmul(xTrainingOnes.getT(), yTraining)

        weight.append(np.matmul(np.linalg.inv(XTX + regParamI), XTy))
        errorsTraining.append(getError(validationSet, ySplit[block], weight[i]))
    print(errorsTraining)
    plt.subplot(222)
    plt.plot(regParam, errorsTraining, 'ro')
    plt.xscale("log")

    bestLambda.append(errorsTraining)

averageLambda = np.matrix(bestLambda).T.A
averageError = []
for i in range(len(averageLambda)):
    average = 0
    for j in range(len(averageLambda[i])):
        average += averageLambda[i][j]
    averageError.append(average / len(averageLambda[i]))

print(averageError)

plt.subplot(221)
plt.plot(regParam, averageError, 'ro')
plt.xscale("log")



plt.show()