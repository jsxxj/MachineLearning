# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 16:47:24 2017

@author: jsx
"""
from numpy import *
import os 

'''图片转换函数，一维矩阵'''
def img2Vector(fileName):
    fr = open(fileName)
    returnMat = zeros((1,1024))
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnMat[0,32*i+j] = int[line[j]]
    return returnMat
    
def classify0(x,dataSet,labels,k):
    m = dataSet.shape[0] #数据样本数
    '''tile()重复输出函数'''
    diffMat = tile(x,(m,1)) - dataSet
    sqDiffMat = diffMat**2
    distance = sqDiffMat**0.5
    sortedDistIndicies = distance.argsort()
    for i in range(k):
        voteIlabel = label[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter[1],reverse=True)
    return sortedClassCount[0][0]
    
def digitClassTest():
    trainingLabels = []
    trainingFiles = os.listdir('trainingDigits')
    m = len(trainingFiles)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNames = trainingFiles[i]
        fileStr = fileNames.split('.')[0]
        classLabel = int(fileStr.split('_')[0])
        trainingLabels.append(classLabel)
        trainingMat[i,:] = img2Vector('trainingDigits/%s' % fileNames)
    testFiles = os.listdir('testDigits')
    errorCount = 0
    mTests = len(testFiles)
    for i in range(mTests):
        fileNames = testFiles[i]
        fileStr = fileNames.split('.')[0]
        classLabel = fileStr.split('_')[0]
        testVector = img2Vector('testDigits/%s' % fileNames)
        classifierResult = classify0(testVector,trainingMat,trainingLabels,3)
        print "the classifier came back is: %d, the real answer is : %d" % (classifierResult,classLabel)
        if (classifierResult != classLabel):
            errorCount += 1
    print "\n the total number of the erros is %d" % errorCount
    print "\n the total error rate is: %f" % (errorCount/float(mTests))

   
