# f(x,y) = k x^2 exp{-xy^2-y^2+2y-4x}, x>0,y in R #

import random,math
 
def gibbs(N=500,thin=1000):
    x=0
    y=0
    print "Iter  x  y"
    for i in range(N):
        for j in range(thin):
            x=random.gammavariate(3,1.0/(y*y+4))
            y=random.gauss(1.0/(x+1),1.0/math.sqrt(2*x+2))
        print i,x,y
 
gibbs()
