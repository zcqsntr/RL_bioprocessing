import numpy as np
import scipy.optimize as spo
from scipy.integrate import odeint
import matplotlib.pyplot as plt



def objective(w):
    return -np.dot(c, w)


c = np.ones((3,1))
A = np.ones((3,3))
b = np.zeros((3,1))

#c = spo.LinearConstraint(A, 0, c)

def xdot(y, t):

    x, S, P = y

    ms = 0.18
    qs_max = 6
    Ks = 0.385
    mp = 46


    vs_up = ms*(qs_max * S)/(S + Ks)
    vs_lo = ms*(qs_max * S)/(S + Ks)



    res = spo.linprog(c, bounds = ((0,1), (0, vs_up), (0, 1)), A_eq = A, b_eq = b)
    w = res["x"]
    vb = w[0]
    vs = w[1]
    vp = w[2]

    dx = vb * x
    dS = vs * x
    dP = mp * vp * x

    return [dx, dS, dP]


if __name__ == "__main__":
    x0 = 10e-3
    S0 = 20
    P0 = 0

    y0 = [x0, S0, P0]

    t = np.linspace(0, 12, 120)

    sol = odeint(xdot, y0, t)

    print(sol)

    plt.plot(sol[:,0], label = 'biomass')
    plt.plot(sol[:,1], label = 'substrate')
    plt.plot(sol[:,2], label = 'product')
    plt.legend()
    plt.show()



