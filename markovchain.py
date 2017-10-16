__author__ = 's1687487'

import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

T = np.array([[0.5,0.5,0,0,0],[0.5,0,0.5,0,0],[0,0.5,0,0.5,0],[0,0,0.5,0,0.5],[0,0,0,0.5,0.5]],dtype=float)
# Tt = np.linalg.matrix_power(T,500)
# eigvalues, eigvectors = np.linalg.eig(Tt)
# print Tt
# print eigvalues
# print eigvectors

# 15/16 Q3-f(ii)

def MCmetrohastings(N=int(1e3),D=5,T=T):
    samples = np.zeros(shape=N+1,dtype=int)

    #pstar_list = np.random.rand(D)
    p_list = 0.2 * np.ones(D)#pstar_list / np.sum(pstar_list)
    cumul_p_list = np.cumsum(p_list)
    samples[0] = D//2
    for n in range(N):
        accept = False
        xcand = np.digitize(np.random.rand(),cumul_p_list)
        xcand = int(xcand)
        samples[n] = int(samples[n])

        # print('hi',xcand,samples[n])
        _N1 = T[xcand,samples[n]]
        _N2 = p_list[xcand]
        _D1 = T[samples[n],xcand]
        _D2 = p_list[samples[n]]
        a = (T[xcand,samples[n]]*p_list[xcand])/(T[samples[n],xcand]*p_list[samples[n]])

        if np.isnan(a):
            pass
        elif a >= 1.:
            accept = True
        else:
            odd = np.random.rand()
            if odd < a:
                accept = True

        if accept:
            samples[n+1] = xcand
        else:
            samples[n+1] = samples[n]
        n += 1

    return samples

xx = MCmetrohastings()
# print xx
vc = np.bincount(xx)
print vc
ax = plt.bar(np.arange(5),vc)
plt.show()

# d = np.diff(xx).min()
# left_of_first_bin = xx.min() - float(d)/2
# right_of_last_bin = xx.max() + float(d)/2
# plt.hist(xx, np.arange(left_of_first_bin, right_of_last_bin + d, d))
# plt.show()