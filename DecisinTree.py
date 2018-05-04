# -*- coding:utf-8 -*-
#西瓜书 决策树的实现
import math
import random
import numpy as np
import operator
import pickle
'''
输入：原始数据集、子数据集（最后一列为类别标签，其他为特征列）
功能：计算原始数据集、子数据集（某一特征取值下对应的数据集）的香农熵
输出：float型数值（数据集的熵值）
'''
def calcShannonEnt(dataset):
    numSample  = len(dataset)
    labelcounts = {}
    for sample in dataset:
        currentlabel = sample[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel]=0
        labelcounts[currentlabel]+=1
    EntD = 0.0
    for key,item in labelcounts.items():
        EntD -= float(item/numSample)* math.log(float(item/numSample),2)
    return EntD

'''
输入：数据集、数据集中的某一特征所在列的索引、该特征某一可能取值（例如，（原始数据集、0,1 ））
功能：取出在该特征取值下的子数据集（子集不包含该特征）
输出：子数据集
'''
def getSubDataset(dataset,colindex,value):
    subDataset = [] # 用于存储子数据集
    for rowVector in dataset:
        if rowVector[colindex]==value:
            subVector = rowVector[:colindex]
            subVector.extend(rowVector[colindex+1:])
            subDataset.append(subVector)
    return subDataset

'''
输入：数据集
功能：选择最优的特征，以便得到最优的子数据集（可简单的理解为特征在决策树中的先后顺序）
      选出最大信息增益的属性，即第几列 （使用信息增益来选取最优属性）
      使用该信息增益来选取最优属性会导致结果偏向 属性取值多 的
输出：最优特征在数据集中的列索引
'''
def information_Gain(dataset):
    numFeature = len(dataset[0])-1
    EntD = calcShannonEnt(dataset)
    GainD = 0.0
    Feature = -1
    #对每个特征计算信息增益
    for i in range(numFeature):
        feat_i_values = [example[i] for example in dataset]#提取每一列
        uniqueValues = set(feat_i_values)
        feat_i_entropy = 0.0
        for value in uniqueValues:
            subDataset = getSubDataset(dataset,i,value)
            prob_i = len(subDataset)/float(len(dataset))
            feat_i_entropy += prob_i*calcShannonEnt(subDataset)
        info_Gain_i = EntD - feat_i_entropy
        if (info_Gain_i> GainD):
            GainD = info_Gain_i
            Feature = i
    return Feature
'''
输入：数据集
功能：选择最优的特征，以便得到最优的子数据集（可简单的理解为特征在决策树中的先后顺序）
      选出最大信息增益的属性，即第几列  （使用增益率来选出最优属性）
      C45决策树 对可取值数目较少的属性有所偏好 
      /采用启发式来选取  从候选属性中选取信息增益高于平均值的，在从中选择增益率最高的 做为最优属性
输出：最优特征在数据集中的列索引
'''
def C45_gain_ratio(dataset):
    numFeature = len(dataset[0]) - 1
    EntD = calcShannonEnt(dataset)
    GainD_ratioMax = 0.0
    Feature = -1
    info_Gain = []
    GainD_ratio = []
    # 对每个特征计算信息增益
    for i in range(numFeature):
        feat_i_values = [example[i] for example in dataset]  # 提取每一列
        uniqueValues = set(feat_i_values)
        feat_i_entropy = 0.0
        IV_i =0.0
        for value in uniqueValues:
            subDataset = getSubDataset(dataset, i, value)
            prob_i = len(subDataset) / float(len(dataset))
            feat_i_entropy += prob_i * calcShannonEnt(subDataset)
            IV_i += -prob_i * math.log(prob_i,2)
        info_Gain_i = EntD - feat_i_entropy
        if IV_i ==0: IV_i= 0.1  #当IV=0是，会报错,设为一个较小的数
        Gain_ratio_i = info_Gain_i/IV_i#相比信息增益增加一个固有值IV
        info_Gain.append(info_Gain_i)
        GainD_ratio.append(Gain_ratio_i)
    info_Gain_ave = float(sum(info_Gain)/len(info_Gain))#信息增益的平均
    #print('信息增益的平均：',info_Gain_ave)
    for i in range(len(info_Gain)):
        if info_Gain[i]>= info_Gain_ave:
            if GainD_ratio[i] > GainD_ratioMax:
                GainD_ratioMax = GainD_ratio[i]
                #print(GainD_ratioMax, i)
                Feature = i
    return Feature
'''
输入：数据集
功能：选择最优的特征，以便得到最优的子数据集（可简单的理解为特征在决策树中的先后顺序）
      CART 选出最大信息增益的属性，即第几列  （使用基尼指数来选出最优属性） 
输出：最优特征在数据集中的列索引
'''
def CART(dataset):
    numFeature = len(dataset[0]) - 1
    GiniD = 1.0
    Feature = -1
    for i in range(numFeature):
        feat_i_values = [example[i] for example in dataset]  # 提取每一列
        uniqueValues = set(feat_i_values)
        Gini_index_i = 0.0
        for value in uniqueValues:
            subDataset = getSubDataset(dataset, i, value)
            prob_i = len(subDataset) / float(len(dataset))
            Gini_index_i += prob_i * Gini(subDataset)
        if Gini_index_i < GiniD:
            GiniD = Gini_index_i
            Feature = i
    return Feature

'''
输入：数据集
功能：计算数据集的基尼值 
输出：数据集的基尼值
'''
def Gini(dataset):
    numSample = len(dataset)
    y = [example[-1] for example in dataset]#提取类别
    classCount ={}
    for class_i in y:
        if class_i not in classCount.keys():
            classCount[class_i]=0
        classCount[class_i] +=1
    counts = [count*count for _,count in classCount.items()]
    Gini = 1 - float(sum(counts)/(numSample*numSample))
    #print(Gini)
    return Gini

'''
输入：子数据集的类别标签列
功能：找出该数据集个数最多的类别(叶节点)
输出：子数据集中个数最多的类别标签
'''
def mostClass_node(classlist):
    classcount = {}
    for class_i in classlist:
        if class_i not in classcount.keys():
            classcount[class_i] = 0
        classcount[class_i] +=1
    sortedclasscount = sorted(classcount.items(),
                              key = operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

'''
输入：数据集，特征标签
功能：创建决策树（直观的理解就是利用上述函数创建一个树形结构）
输出：决策树（用嵌套的字典表示）
'''
def creatTree(dataset,labels):
    featurelabels =labels
    classlist = [example[-1] for example in dataset]
    #判断传入的dataset是否只有一个类别
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    #判断是否遍历所有的特征
    if len(dataset[0])==1:
        return mostClass_node(classlist)
    #找出最好的特征或属性（最大的信息增益的属性）
    #bestFeature = information_Gain(dataset) #采用信息增益
    #bestFeature = C45_gain_ratio(dataset)#采用信息增益率
    bestFeature = CART(dataset) #采用CART基尼指数
    bestFeatlabel = featurelabels[bestFeature]
    #搭建树结构
    myTree = {bestFeatlabel:{}}
    featurelabels.pop(bestFeature)#删去这个最优属性
    bestFeatValues = [example[bestFeature] for example in dataset]
    uniqueBestFeatValues = set(bestFeatValues)#最优属性的取值个数
    for value in uniqueBestFeatValues:
        subDataset = getSubDataset(dataset,bestFeature,value)
        sublabels = labels[:]
        myTree[bestFeatlabel][value] = creatTree(subDataset,sublabels)
    return myTree

'''
输入：测试特征数据
功能：调用训练决策树对测试数据打上类别标签
输出：测试特征数据所属类别
'''
def classify(inputTree,featlabels,testFeatValue):
    firstStr= list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featlabels.index(firstStr)
    for firstStr_value in secondDict.keys():
        if testFeatValue[featIndex] == firstStr_value:
            if type(secondDict[firstStr_value]).__name__ == 'dict':
                classLabel = classify(secondDict[firstStr_value],featlabels,testFeatValue)
            else: classLabel = secondDict[firstStr_value]
    return classLabel
'''
输入：训练树，存储的文件名
功能：训练树的存储
输出：
'''
def storeTree(trainTree,filename):
    fw = open(filename,'w')
    pickle.dump(trainTree)
    fw.close()

def gradTree(filename):
    fr = open(filename)
    return pickle.load(fr)

#预剪枝
def pre_pruning():
    pass
if __name__ == '__main__':
    x = [[0,0,0,0,0,0,1],
     [1,0,1,0,0,0,1],
     [1,0,0,0,0,0,1],
     [0,0,1,0,0,0,1],
     [2,0,0,0,0,0,1],
     [0,1,0,0,1,1,1],
     [1,0,0,1,1,1,1],
     [1,1,0,0,1,0,1],
     [1,1,1,1,1,0,0],
     [0,2,2,0,2,1,0],
     [2,2,2,2,2,0,0],
     [2,0,0,2,2,1,0],
     [0,1,0,1,0,1,0],
     [2,1,1,1,0,0,0],
     [1,1,0,0,1,1,0],
     [2,0,0,2,2,0,0],
     [0,0,1,1,1,0,0]]
    featurelabels = ['色泽','根蒂','敲声','纹理','脐部','触感']
    labels = ['色泽','根蒂','敲声','纹理','脐部','触感']#python 函数是引用传递，前面函数会删去labels
    trainTree = creatTree(x,featurelabels)
    print(trainTree)
    classlabel = classify(trainTree,labels,[0,1,0,0,1,1])
    print(classlabel)
