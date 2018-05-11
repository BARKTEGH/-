# -*- coding:utf-8 -*-
'''
/@2018-05-08
/@author:BARKTEGH
隐马尔科夫模型的 预测问题
近似计算： 把Yt[i]的最大值作为t时刻最有可能的状态
    定义delta为时刻t状态为i 的所有单个路径中概率最大值
    phi 为取得delata[t][i]最大值的前一个节点
    每部迭代 取得每个时刻t的 前一个节点 使得t时刻为i状态概率最大，
    找出最大概率的节点，反向迭代，找出前一个节点，。。，最终认为这条路径就是最优路径
维特比算法： 动态规划解来解决隐马尔可夫预测问题
'''
import numpy as np
import matplotlib.pyplot as plt

def Viterbi_algorithm(A,B,pi,o):
    #初始化
    N = len(A[0])
    T = len(o)
    delta =[([0] * N) for i in range(N)]
    phi = [([0] * N) for i in range(N)]
    i_bestroad = []
    for i in range(N):
        delta[0][i] = pi[i]*B[i][o[0]]
        phi[0][i] = 0
    for t in range(1,T):
        for i in range(N):
            aa=0
            j_t = 0
            for j in range(N):
                if delta[t-1][j]*A[j][i] >aa:
                    aa = delta[t-1][j]*A[j][i]#找出由j节点转移到i点的最大值
                    j_t = j #j节点 ，认为是当前t时刻可能的最大概率路径的前一个节点
            delta[t][i] = aa * B[i][o[t]]
            phi[t][i] = j_t
    print(delta)
    print(phi)
    P = max(delta[T-1])
    iT = delta[T-1].index(max(delta[T-1]))
    i_bestroad.append(iT)
    for t in reversed(range(1,T)):
        i_t = phi[t][i_bestroad[0]]
        i_bestroad.insert(0,i_t)
    print(i_bestroad)
    return i_bestroad,delta,phi

def plot_allroad(bestroad,delta,phi):
    T = len(delta)
    N = len(delta[0])
    x1 = [([i] * N) for i in range(N)]
    y1 = [[i  for i in range(N)]] *(N)
    x3=[]
    y3=[]
    for t in range(T):
        x3.append(t)
        y3.append(bestroad[t])
    for t in range(T-1):
        for i in range(N):
            plt.plot([t,t+1],[phi[t+1][i],i],'blue')
    plt.plot(x1,y1,'ro',label='node')
    plt.plot(x3,y3,'red',label='bestroad')
    plt.xlabel('Time')
    plt.ylabel('Stage')
    plt.legend()
    plt.title('ALL max prob road and best road')
    plt.show()

def jinsijisuan(A,B,pi,o):
    pass

if __name__ =='__main__':
    A= np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    B= np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
    pi = np.array([0.2,0.4,0.4])
    o = [0,1,0]
    bestroad, delta, phi=Viterbi_algorithm(A,B,pi,o)
    plot_allroad(bestroad,delta,phi)