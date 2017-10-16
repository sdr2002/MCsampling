__author__ = 's1687487'

import numpy as np
import matplotlib.pyplot as plt

# Slice sampling for 14/15 MLPR Q3-(b)
# Theta is x in this code
p = lambda x: 2.*x if (x>=0.) and (x<=2.) else 0.

def slicesampling(N=int(1e5)):

    samples = np.zeros(N)
    samples[0] = 0.1

    for n in range(1,N):
        _high = p(samples[n-1])
        ycand = np.random.uniform(low=0.,high=p(samples[n-1]))
        xl, xr = create_slice(samples[n-1],ycand)
        while True:
            xcand = np.random.uniform(low=xl,high=xr)
            if ycand < p(xcand):
                samples[n] = xcand
                break
            else:
                modify_slice(samples[n],xcand,xl,xr) # Shrink the slice

    return samples

def create_slice(xs,ycand,w=0.1):
    r = abs(np.random.rand())
    xl = xs - r*w
    xr = xs + r*w
    while p(xl) > ycand:
        xl -= w
    while p(xr) > ycand:
        xr += w

    return [xl, xr]

def modify_slice(xs,xcand,xl,xr):
    if xcand >= xs:
        xr = xcand
    elif xcand <= xs:
        xl = xcand
    else:
        raise
    return [xl, xr]

xx = slicesampling()
n, bins, patches = plt.hist(xx, 100)
plt.show()