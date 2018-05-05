# -*- coding:utf-8 -*-
#2018-05-05
#提取Decision部分 ,生成一颗完整的ID3决策树
#没有采用阈值
#ID3算法采用信息增益，但是信息增益偏爱取值较多的属性，导致会完全展开
import math
import random
import numpy as np
import operator
import pickle
from DecisionTree import calcShannonEnt,getSubDataset,information_Gain,mostClass_node,classify,storeTree,gradTree

'''
输入：数据集，特征标签
功能：创建决策树（直观的理解就是利用上述函数创建一个树形结构）
输出：决策树（用嵌套的字典表示）
'''
def creatTree_ID3(dataset,labels):
    featurelabels =labels
    classlist = [example[-1] for example in dataset]
    #判断传入的dataset是否只有一个类别
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    #判断是否遍历所有的特征
    if len(dataset[0])==1:
        return mostClass_node(classlist)
    #找出最好的特征或属性（最大的信息增益的属性）
    bestFeature = information_Gain(dataset) #采用信息增益
    bestFeatlabel = featurelabels[bestFeature]
    #搭建树结构
    myTree = {bestFeatlabel:{}}
    featurelabels.pop(bestFeature)#删去这个最优属性
    bestFeatValues = [example[bestFeature] for example in dataset]
    uniqueBestFeatValues = set(bestFeatValues)#最优属性的取值个数
    for value in uniqueBestFeatValues:
        subDataset = getSubDataset(dataset,bestFeature,value)
        sublabels = labels[:]
        myTree[bestFeatlabel][value] = creatTree_ID3(subDataset,sublabels)
    return myTree

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
    x2 = [0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719]
    featurelabels = ['色泽','根蒂','敲声','纹理','脐部','触感','密度']
    labels = ['色泽','根蒂','敲声','纹理','脐部','触感','密度']#python 函数是引用传递，前面函数会删去labels
    trainTree = creatTree_ID3(x,featurelabels)
    print(trainTree)
    classlabel = classify(trainTree,labels,[0,1,0,0,1,1,0])
    print(classlabel)