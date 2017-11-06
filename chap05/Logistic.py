# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 22:53:24 2017

@author: jsx
"""
from numpy import *
import matplotlib.pyplot as plt

'''定义文件加载函数'''
def loadDataSet():
    dataMat = []
    labelMat = []
    f = open('testSet.txt','r')
    for line in f.readlines():
        line = line.strip().split()
        dataMat.append([1.0,float(line[0]),float(line[1])])
        labelMat.append(int(line[2]))
    return dataMat,labelMat
    
def sigmoid(x):
    return 1.0/(1+exp(-x))

def gradAscent(dataMat,classLabel):
    '''
    梯度上升函数；输入为数据矩阵，数据类别标签
    '''
    #convert to the numpy matrix
    dataMatrix = mat(dataMat)
    '''convert to the numpy matrix'''
    labelMatrix = mat(classLabel).transpose() 
    m,n = dataMatrix.shape
    '''步长'''
    alpha = 0.001   
    '''最大迭代次数'''
    maxCycles = 500 
    weights = ones((n,1))
    for k in range(maxCycles):
        '''h表示一个列向量'''
        h = sigmoid(dataMatrix*weights)  
        error = labelMatrix - h
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights
    
#随机梯度上升
'''
每次只需要计算单个样本 来更新回归系数
'''
def stocGradAscent(dataMat,classLabel):
    m,n = dataMat.shape
    alpha = 0.01
    weights = ones((n,1))
    for i in range(m):
        h = sigmoid(sum(dataMat[i]*weights))
        error = classLabel[i] - h
        weights = weights + alpha*error*dataMat[i]
    return weights
    
'''改进的梯度上升算法'''
def stocGradAscent2(dataMat,classLabel,numInter=150):
    m,n = dataMat.shape
    weights = ones((n,1))
    for j in range(numInter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)*0.01
            randomIndex = int(random.uniform(0,len(dataIndex)))  #随机选择样本更新回归系数
            h = sigmoid(sum(dataMat[randomIndex]*weights))
            error = classLabel[randomIndex] - h
            weights = weights + alpha*error*dataMat[randomIndex]
            del(dataMat[randomIndex]) #从列表中删除这个数值
    return weights
    
def plotBestFit():
    dataMat,labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    dataArr = array(dataMat)
    n = dataArr.shape[0]
    xcord1 = []
    ycord1 = []                                                                                                                                                                                                                                                                            
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2');
    plt.show()


'''采用逻辑回归预测疝气病马的死亡率'''

def classifyVector(inx,weights):
    prob = sigmoid(sum(intx*weights))
    if prob>0.5:
        return 1
    else:
        return 0
        
def colicTets():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeights = stocGradAscent1(array(trainingSet,trainingLabels,500))
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainingWeights)) != int(currLine[21]):
            errorCount += 1
        errorRate = (float(errorCount)/numTestVec)
        print 'the error rate of the test is : %f' % errorRate
        
        return errorRate
        
def multiTest():
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += colicTests()
    print 'after %d iterations the average error rate is: %f' % (numTests,errorSum/float(numTests))
    
    
if __name__ == '__main__':
    plotBestFit()

  
