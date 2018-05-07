# -*- coding:utf-8 -*-
# 2018-05-06
#提取Decision部分 ,生成一颗完整的CART决策树

import math
import random
import numpy as np
import operator
import pickle

from DecisionTree import getSubDataset,storeTree,gradTree

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

def getSubdataset_no(dataset,colindex,value):
    subDataset_no = []  # 用于存储子数据集
    for rowVector in dataset:
        if rowVector[colindex] != value:
            subVector = rowVector[:colindex]
            subVector.extend(rowVector[colindex + 1:])
            subDataset_no.append(subVector)
    return subDataset_no
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
    BESTvalue = 0
    for i in range(numFeature):
        feat_i_values = [example[i] for example in dataset]  # 提取每一列
        uniqueValues = set(feat_i_values)
        for value in uniqueValues:
            subDataset = getSubDataset(dataset, i, value)
            subDataset_no = getSubdataset_no(dataset,i,value)
            prob_i = len(subDataset) / float(len(dataset))
            prob_i_no = len(subDataset_no)/float(len(dataset))
            if len(subDataset_no)==0:
                Gini_i_value = prob_i * Gini(subDataset) +0.8
            else:
                Gini_i_value = prob_i * Gini(subDataset)+prob_i_no*Gini(subDataset_no)
            if Gini_i_value < GiniD:
                GiniD = Gini_i_value
                Feature = i
                BESTvalue = value
    return Feature,BESTvalue
'''
输入：数据集，特征标签
功能：创建决策树（直观的理解就是利用上述函数创建一个树形结构）
输出：决策树（用嵌套的字典表示）
'''
def creatTree_CART(dataset,labels):
    featurelabels =labels
    classlist = [example[-1] for example in dataset]
    #判断传入的dataset是否只有一个类别
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    #判断是否遍历所有的特征
    if len(dataset[0])==1:
        return mostClass_node(classlist)
    bestFeature,BESTvalue = CART(dataset) #采用CART基尼指数
    bestFeatlabel = featurelabels[bestFeature]
    #搭建树结构
    myTree = {bestFeatlabel:{}}
    featurelabels.pop(bestFeature)#删去这个最优属性
    uniqueBestFeatValues = [BESTvalue,False] #最优属性的取值个数
    subDataset = getSubDataset(dataset,bestFeature,BESTvalue)
    if len(subDataset)!=0:
        sublabels = labels[:]
        myTree[bestFeatlabel][BESTvalue] = creatTree_CART(subDataset,sublabels)
    else:
        myTree[bestFeatlabel][BESTvalue] = 'NULL'
    subDataset_no = getSubdataset_no(dataset, bestFeature, BESTvalue)
    if len(subDataset_no)!=0:
        sublabels = labels[:]
        myTree[bestFeatlabel][False] = creatTree_CART(subDataset_no, sublabels)
    else:
        myTree[bestFeatlabel][False] = 'NULL'
    return myTree

def pruning(myTree):
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
    trainTree = creatTree_CART(x,featurelabels)
    print(trainTree)
