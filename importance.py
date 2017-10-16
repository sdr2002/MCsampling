# coding=utf-8
__author__ = 's1687487'

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

f = lambda x: 0. if (x < 0.) or (x > np.pi) else np.sin(x)*np.cos(np.exp(x))**2
qstar = lambda x,mu,sigma: np.exp(-(x-mu)**2/(2.*sigma**2))/np.sqrt(2.*np.pi*sigma**2)

def importance(N=int(1e4)): # Iain's tute 5
    x_list = np.array([])
    wstar_list = np.array([])
    pstar_x_list = np.array([])

    for n in range(N):
        x = 2.*np.random.randn() + 1.5

        x_list = np.append(x_list, x)
        qstar_x = qstar(x,mu = 1.5, sigma = 2.)
        pstar_x = f(x)
        pstar_x_list = np.append(pstar_x_list, pstar_x)

        wstar = pstar_x/qstar_x
        wstar_list = np.append(wstar_list, wstar)
    w_list = wstar_list / np.sum(wstar_list)

    estimate_mean = np.sum(x_list * w_list)
    estimate_Z = 1/float(N)*np.sum(wstar_list)
    return w_list, x_list, x_list * w_list, estimate_mean, estimate_Z

# tt = 6;
# xx = abs(randn(S, 1)) + tt;
# q_x = 2*exp(-(xx-tt).ˆ2/2)/sqrt(2*pi);
# pstar_x = exp(-xx.ˆ2/2)/sqrt(2*pi);
# wstar = (pstar_x./q_x); % unnormalized weights. pstar_x *not* normalized!
# ww = wstar/sum(wstar);
# est = sum(ww.*xx)

ww, xx, wx, est_mean, est_Z = importance()
# plt.scatter(xx,wx)
# # plt.xlim(-1.,4.)
# plt.ylim(-0.0001,0.0002)
# # n2, bins2, patches2 = plt.hist(wx, 100)
# plt.show()
print est_mean, est_Z

real_mean = integrate.quad(lambda x: x*f(x), 0., np.pi)
real_Z = integrate.quad(lambda x: f(x), 0., np.pi)
print real_mean, real_Z
sum_q = integrate.quad(lambda x: qstar(x, mu = 1.5, sigma = 2.), -100., 100.)
print sum_q