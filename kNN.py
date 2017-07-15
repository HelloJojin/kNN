# -*- coding=utf-8 -*-
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
import numpy as np
import os

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])#  返回B
    #group = array ([[2.0, 1.1], [3.0, 1.0], [0, 0], [0, 0.1]])  #  返回A
    labels = ['A', 'B', 'A', 'B']
    return group, labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] # dataSetSize = 900; 返回的是group矩阵的行数。
    print('dataSetSize------------------',dataSetSize)
    # distance calculate
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    print('tile(inX,(dataSetSize,1))-dataSet', tile(inX, (dataSetSize, 1)) -dataSet)
    print('diffMat',diffMat)
    sqDiffMat = diffMat**2    #平方将矩阵的元素转化成正值。
    print('diffMat**2:', sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)#将行向量相加
    print('sqDistances=', sqDistances)
    distances = sqDistances**0.5
    print('distances=', distances)
    sortedDistIndicies = distances.argsort() # argsort返回的是排序完成后的索引值。
    print('sortedDistIndicies=', sortedDistIndicies)



    for i in sortedDistIndicies:
        print 'sortedDistIndicies [%d]' % int(i), sortedDistIndicies[i]

    classCount={}
    #  选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 0->2->A,1->3->B,2->1->B
        print 'voteIlabel', voteIlabel
        # 下面这一步做了一个很重要的工作就是出现相同分类的进行合并
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # A:1,B:2
    print('classCount:', classCount)
    #  classCount.interitems() 返回列表的迭代器, key=operator.intemgetter(1)表示按照classCount中第一域的元素排序,也就是按照字母A,B来排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    print 'sortedClassCount', sortedClassCount
    print 'sortedClassCount[0][0]', sortedClassCount[0][0]
    return sortedClassCount[0][0]

group, labels = createDataSet()
print('group',shape(group)) # 返回矩阵的形状
print('group-0',group.shape[0])  # 返回矩阵group的行数。
print('group-1',group.shape[1])  # 返回矩阵group的列数
x = classify0([0, 0], group, labels, 3)
print 'kNN is ', x


dict = {'Name': 'Zara', 'Age': 27}
print "Value : %s" % dict.get('Age')
print "Value : %s" % dict.get('Sex', 'nihao')
x = random.rand(4,3)
print(shape(x))
y = mat(x)
yy=y.I
print(shape(yy))
z = yy.shape[0]
print(z)


def file2matrix(filename):
    fr = open(filename)
    print 'fr :' , fr
    array0Lines = fr.readlines()
    print 'array0Lines', array0Lines
    number0fLines = len((array0Lines)) #  得到文件行数
    print 'number0fLines:', number0fLines
    returnMat = zeros((number0fLines, 3))
    print 'returnMat:', np.shape(returnMat)
    classLabelVector = []
    index = 0
    print '------------------------'
    # 解析文件数据到列表中
    for line in array0Lines:
        line = line.strip() #strip() 默认删除空白字符，\r 回车键,\t,\n
        # print 'line: ', line  # 打印的效果是这种格式 line:  40920	8.326976	0.953952	3
        listFromLine = line.split('\t')
        # print 'listFromLine :', listFromLine
        returnMat[index,:] = listFromLine[0:3]
        #print 'returnMat', returnMat
        classLabelVector.append(int(listFromLine[-1]))
        if len(classLabelVector) == 1000:
            print 'classLabelVector:', classLabelVector
        index += 1
    return returnMat, classLabelVector

datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# print datingDataMat
# print datingLabels
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
           15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

#  归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 从列中选取最小值
    print 'minVals ', minVals # minVals  [ 0.        0.        0.001156]
    maxVals = dataSet.max(0)
    print 'maxVals ', maxVals #maxVals  [  9.12730000e+04   2.09193490e+01   1.69551700e+00]
    ranges = maxVals - minVals
    print 'range:', ranges  #range: [  9.12730000e+04   2.09193490e+01   1.69436100e+00]
    normDataSet = zeros(shape(dataSet))
    print 'shape(dataSet):', shape(dataSet) #shape(dataSet): (1000L, 3L)
    m = dataSet.shape[0] # m 就是矩阵dataSset的行数。m = 1000
    normDataSet = dataSet - tile(minVals,(m,1)) #tile部分构建了一个每一行都是最小值的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))  #  特征值相除

    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)
print 'normMat:', normMat
print 'ranges:', ranges
print 'minVals:', minVals

# 2-4 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
                            datingLabels[numTestVecs:m], 3)
        #print "the classifier came back with: %d, the real answer is : %d"\
                              # % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

datingClassTest()

s = '\t40920\t8.326976\t0.953952\t3\n'
print 'sssssssss', s
a=s.strip()
print 'aaaaaaaaa', a #40920	8.326976    0.953952    3
b = a.split('\t')
print 'bbbbbbbb', b

plt.figure(1)

# 2-5约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    ffMiles= float(raw_input('frequent filer miles earned per yeaer?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print 'You will probably like this person: ', resultList[classifierResult -1]

#classifyPerson() # 约会测试函数暂时先不调用

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        print 'lineStr {}, {}'.format(i, lineStr)
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
testVector = img2vector('digits/testDigits/0_13.txt')
#print 'testVector:', testVector.size  # 1024 = 32 * 32
x = testVector[0,32:63]
#print 'testVector:', testVector
print x
'''
x = [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  
      0.  0.  1.  1.  1.  1.  1.  1.  1.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
      0000000000
      0011111110
      000000000000
    经过比对，和文件中的完全一样
'''
# 2.6 handwriting 手写识别系统的测试代码

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits') # 获取目录内容,返回的是一个关于该目录下文件名称的一个列表
    print 'trainingFileList:', trainingFileList
    m = len(trainingFileList)
    print 'm:', m # m = 1934
    trainingMat = zeros((m, 1024))  # 生成一个1934*1024的二维数组
    for i in range(m):
        # 以下三行从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr= fileNameStr.split('.'[0]) # fileStr: ['0_0', 'txt']
        if i == 0 :
            print 'fileStr:', fileStr
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        # print 'the classifer came back with: %d, the real answer is :%d' % (classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1.0
    print '\nthe total number of errors is: %d' % errorCount
    print '\nthe total error rate is: %f' % (errorCount/float(mTest))
handwritingClassTest()
