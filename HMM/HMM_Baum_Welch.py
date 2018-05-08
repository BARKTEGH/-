# -*- coding:utf-8 -*-
'''
/@2018-05-08
/@author:BARKTEGH
隐马尔科夫模型的 学习问题
监督学习 ：给定观测序列和对应的状态序列
非监督学习：EM算法 给定S个T长度的观测序列
'''
import numpy as np


def forward(A, B, pi, o):
    '''
    :param A: 转移矩阵
    :param B: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param o: 观测序列 规定为（0,1,2,3,4）第0种状态
    :return:  alphaT T个时刻的alpha值
    '''
    T = len(o)
    N = len(A)
    alphaT = []
    alpha = []
    # 初始化
    for i in range(N):
        alpha.append(pi[i] * B[i][o[0]])
    alphaT.append(alpha)
    for t in range(1, N):
        temp = []  # 临时存储alpha i，防止在计算 i+2就是用更新后的alpha值
        for i in range(N):
            ci = 0.0
            for j in range(N):
                ci += alpha[j] * A[j][i]
            temp.append(ci * B[i][o[t]])
        alphaT.append(temp)
        alpha = temp
    #pro = sum(alpha)
    return alphaT

def backward(A,B,pi,o):
    '''
    :param A: 转移矩阵
    :param B: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param o: 观测序列 规定为（0,1,2,3,4）第0种状态
    :return:  betaT
    '''
    betaT =[]
    N = len(A)
    T = len(o)
    #初始化
    beta = np.ones(N)
    betaT.append([1.0,1.0,1.0])
    for t in reversed(range(1,T)):
        temp =[]
        for i in range(N):
            ci =0.0
            for j in range(N):
                ci += A[i][j] * B[j][o[t]] * beta[j]
            temp.append(ci)
        beta = temp
        betaT.append(beta)
    #pro = sum([pi[i]*B[i][o[0]]*beta[i]  for i in range(N)])
    return betaT

def Baum_welch(o,N,m):
    T = len(o[0])
    m = m
    A = np.zeros((N,N))
    B = np.zeros((N,m))
    pi = np.zeros(N)
    alphaT = forward(A,B,pi,o)
    betaT = backward(A,B,pi,o)
    done = True
    while done:
        gama =[]
        sigama =[]
        for t in range(T):
            dt =0.0
            for i in range(N):
                ci =0.0
                for j in range(N):
                    sigama[t][i][j] = alphaT[t][i]*A[i][j]*B[j][o[t]]*betaT[T-t-1][j]
                    ci += alphaT[t][j]*betaT[T-1-t][j]
                    dt += alphaT[t][i]*A[i][j]*B[j][o[t]]*betaT[T-t-1][j]
                gama[t][i] = alphaT[t][i] *betaT[T-t-1]/ci
            sigama[t] = sigama/dt
        for i in range(N):
            for j in range(N):
                yy =0.0
                ee =0.0
                for t in range(T-1):
                    ee += sigama[t][i][j]
                    yy += gama[t][i]
                A[i][j]= ee/yy
            for k in range(m):
                yy=0.0
                ytt =0.0
                for t in range(T):
                    yy += gama[t][i]
                    pass #不理解ot=Vk的含义
                B[i][k] =




if __name__ =='__main__':
    A= np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    B= np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
    pi = np.array([0.2,0.4,0.4])
    o = [0,1,0]
    print(backward(A,B,pi,o))
