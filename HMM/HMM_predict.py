# -*- coding:utf-8 -*-
'''
/@2018-05-08
/@author:BARKTEGH
隐马尔科夫模型的 预测问题
近似计算： 把Yt[i]的最大值作为t时刻最有可能的状态
维特比算法： 动态规划解来解决隐马尔可夫预测问题
'''
import numpy as np
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
                    aa = delta[t-1][j]*A[j][i]
                    j_t = j
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
    return i_bestroad

def jinsijisuan(A,B,pi,o):
    pass

if __name__ =='__main__':
    A= np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    B= np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
    pi = np.array([0.2,0.4,0.4])
    o = [0,1,0]
    Viterbi_algorithm(A,B,pi,o)
