import numpy as np
import matplotlib.pyplot as plt

logptilde = lambda x: np.log(np.exp(-0.5*np.dot(x+2.,x+2.))+np.exp(-0.5*np.dot(x-6.,x-6.)))

def hamiltonian(init=0., iters=int(1e4), Tau=10, epsilon=5e-2): # Mackay's book p.388
    D = len(init) if not isinstance(init, float) else int(1)
    samples = np.zeros(shape=(D, iters))

    findE = lambda x: -logptilde(x)
    gradE = lambda x: 1./np.exp(logptilde(x)) * (-0.5*2*(x+2)*np.exp(-0.5*np.dot(x+2.,x+2.))-0.5*2*(x-2)*np.exp(-0.5*np.dot(x-7.,x-7.))) # = dE/dx (you can do this numerically by having x & dx as arguments)

    E = findE(init) ; # set objective function using initial x
    _E = E
    g = gradE(init) ; # set gradient too
    _g = g

    x = init
    for i in range(iters): # loop L times
        #Gaussian Propose of p
        p = np.random.randn(D) # initial momentum is Normal(0,1)
        _p = p
        H = np.dot(p,p) / 2. + E # evaluate H(x,p)
        xnew = x
        gnew = g

        for tau in range(Tau): # make Tau `leapfrog' steps: Walking along the Hamiltonian dynamics
            p = p - epsilon * gnew / 2. # make half-step in p
            xnew = xnew + epsilon * p # make step in x
            gnew = gradE( xnew ) # find new gradient
            p = p - epsilon * gnew / 2. # make half-step in p

        Enew = findE(xnew) # find new value of H
        Hnew = np.dot(p,p) / 2. + Enew

        dH = Hnew - H # Decide whether to accept
        if dH < 0:
            accept = True
        elif np.random.rand() < np.exp(-dH):
            accept = True
        else:
            accept = False

        if accept:
            g = gnew ; x = xnew ; E = Enew ;
        samples[:,i] = x

    return samples
xx = hamiltonian()

n, bins, patches = plt.hist(xx[0], 100)
plt.show()