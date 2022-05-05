
from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt


def g_d_alpha(d, alpha):
    g = lambda k: ((k+1)**(2*d+1) - k**(d+1)*(k+2)**d + alpha*((k+1)**(d+1) - k**(d+1))) / \
                  ((1+alpha)*(k+1)**(d+1) - (1+alpha)*k**(d+1) - (k+2)**d + (k+1)**d)
    return g


def f_d_alpha(d, alpha):
    f = lambda x: x**(d+1) - (x+1)**d + alpha*(x**(d+1)-1)
    return f


def q_d_alpha(d, alpha):
    mu = 0.5
    q = lambda x: ((x+1)**d - mu*x**(d+1) + alpha) / (1 - mu + alpha)
    return q


d = 5
alphas = np.linspace(0, 1, 101)

PoA_unweighted = []
PoA_weighted = []



for alpha in alphas:
    f = f_d_alpha(d, alpha)
    g = g_d_alpha(d, alpha)
    q = q_d_alpha(d, alpha)
    Psi = newton(f, 3**d)
    k = np.floor(Psi)
    PoA_unweighted.append(g(k))
    PoA_weighted.append(q(Psi))
    
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})
    
plt.plot(alphas, PoA_unweighted, label="unweighted")
plt.plot(alphas, PoA_weighted, label="weighted")
plt.grid(visible=True)

plt.legend()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"PoA upper-bound")
plt.title("Price of Anarchy Bounds for Degree {} Polynomial Cost Functions".format(d))

plt.show()

# plt.savefig("PoA_degree{}.pdf".format(d))
# plt.savefig("PoA_degree{}.png".format(d), dpi=300)


