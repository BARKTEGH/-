# -*- coding:utf-8 -*-
'''
/@2018-05-08
/@author:BARKTEGH
隐马尔科夫模型的 概率计算问题
给定模型lamda =(A,B,pi),和观测序列0，计算在模型lamda下观测序列出现的概率
'''
import numpy as np
'''
实质时通过在第i步的所有状态（隐藏）和已观测状态的联合概率转移到i+1步的所有状态（隐藏）与已观测状态的概率
一步步从前向后推到，最终得到 观测序列出现的概率
'''
def forward(A,B,pi,o):
    '''
    :param A: 转移矩阵
    :param B: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param o: 观测序列 规定为（0,1,2,3,4）第0种状态
    :return:  Prob 概率
    '''
    print('forward')
    T = len(o)
    N = len(A)
    alpha = []
    #初始化
    for i in range(N):
        alpha.append(pi[i] * B[i][o[0]])
    print('alpha1=',alpha)
    for t in range(1,N):
        temp =[]#临时存储alpha i，防止在计算 i+2就是用更新后的alpha值
        for i in range(N):
            ci=0.0
            for j in range(N):
                ci += alpha[j]*A[j][i]
            temp.append(ci * B[i][o[t]])
        alpha = temp
        print('alpha{}='.format(t+1),alpha)
    pro = sum(alpha)
    return pro

'''
与前向传播类似，不过是反向推导
'''
def backward(A,B,pi,o):
    '''
    :param A: 转移矩阵
    :param B: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param o: 观测序列 规定为（0,1,2,3,4）第0种状态
    :return:  Prob 概率
    '''
    print('backward:')
    N = len(A)
    T = len(o)
    #初始化
    beta = np.ones(N)
    print('alpha{}='.format(N),beta)
    for t in reversed(range(1,T)):
        temp =[]
        for i in range(N):
            ci =0.0
            for j in range(N):
                ci += A[i][j] * B[j][o[t]] * beta[j]
            temp.append(ci)
        beta = temp
        print('alpha{}='.format(t),beta)
    pro = sum([pi[i]*B[i][o[0]]*beta[i]  for i in range(N)])
    return pro


if __name__ =='__main__':
    A= np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    B= np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
    pi = np.array([0.2,0.4,0.4])
    o = [0,1,0]
    print(forward(A,B,pi,o))
    print(backward(A,B,pi,o))