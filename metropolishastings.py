import numpy as np
import matplotlib.pyplot as plt

logptilde = lambda x: np.log(np.exp(-0.5*np.dot(x+2.,x+2.))+np.exp(-0.5*np.dot(x-6.,x-6.)))

def metropolishasting(init = -5., iters = int(1e4), sigma = 1.):
    D = len(init) if not isinstance(init, float) else int(1)
    samples = np.zeros(shape=(D, iters))

    state= init
    Lp_state = logptilde(state)
    for i in range(iters):
        # Gaussian Propose
        proposal = state + sigma*np.random.randn(D)
        Lp_proposal = logptilde(proposal)

        loga = Lp_proposal - Lp_state
        logr = np.log(np.random.rand())
        if logr < loga:
            state = proposal
            Lp_state = Lp_proposal
        samples[:,i] = state

    return samples

xx = metropolishasting()

n, bins, patches = plt.hist(xx[0], 100)
plt.show()