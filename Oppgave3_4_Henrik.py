# %%
#
#  Fra oppgave 3 a)
#

import numpy as np
import matplotlib.pyplot as plt


def analytisk(x):
    return -(1 / np.pi**2) * np.cos(np.pi * x) + x + (1 - (1 / np.pi**2))


#
#   Oppgave 3 b)
#   Finne numerisk løsning

#
#    Formel som skal settes opp:
#   (u[:-2] - 2*u[1:-1] + u[2:]) / dx**2 = f(x)
#   Hvor f(x) = cos(pi*x)


#    Setter opp kjente verdier
m = 40  # antall indre punkter
X = np.linspace(-1, 1, m + 2)  # inkluderer randpunktene
dx = X[1] - X[0]  # steglengde i rom
f = np.cos(np.pi * X[1:-1])


#   Lager matrise u hvor som viser endelig løsning med randbetingelser
u = np.zeros(m + 2)
u[0] = 0  # (Dirichlet-)randbetingelsene for oppgaven
u[-1] = 2  # (Dirichlet-)randbetingelsene for oppgaven


#   Lager matrisen L
#   Brunker np.diag for å lage diagonalmatrise
L = (1 / dx**2) * (
    np.diag(m * [-2], 0) + np.diag((m - 1) * [1], 1) + np.diag((m - 1) * [1], -1)
)

#   Lager matrise b
#   Må legge inn for f(x) i b
b = f
b[0] = -u[0] / (dx**2)
b[-1] = -u[-1] / (dx**2)


#   Ligningen som skal løses nå er L @ u[1:-1] = b
u[1:-1] = np.linalg.solve(L, b)


#   Plotting av løsning
#
plt.figure(figsize=(8, 5))

#   Lager plot av løsning i b)
plt.plot(X, u, label="Numerisk løsning b)", linewidth=2)

#   Lager så plot av løsning i a)
analytisk_løsning = analytisk(X)
plt.plot(X, analytisk_løsning, label="Analytisk løsning a)", linestyle="--")

#   Legger til labels og legends
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Oppgave 3: Poissonligning 1D - analytisk og numerisk")
plt.grid(True)
plt.legend()

plt.show()


# %%
#   Oppgave 4
#   Løser varmelikning med forlengs euler
#

# import numpy as np
# import matplotlib.pyplot as plt

# m = 20  # antall indre punkter
# x = np.linspace(-1, 1, m + 2)  # Romgitter inkl. randpunktene x=-1 og x=1
# h = x[1]- x

c = 1  # Fysiske parameter (varmeledningsevne)
a = 0  # Randbetingelse u(-1)=a
B = 2  # Randbetingelse u(1)=b
A = (c**2) * L  # fordi varmeligningen er ut=c**2*uxx + f

#   Høyre side og randbidrag
# Viktig: For ut = uxx - f(x) bruker vi g(u)=A u - F med F = f + randbidrag
F = f
F[0] -= (c**2) * a / dx**2  # bidrag fra venstre rand
F[-1] -= (c**2) * B / dx**2  # bidrag fra høyre rand


#  --- Lager en funksjon for forlengs Euler i tid ---
def euler(g, x0, t0, t1, N):
    t = np.linspace(t0, t1, N)
    dt = t[1] - t[0]
    out = np.zeros((N, x0.size))
    out[0, :] = x0

    for n in range(N - 1):
        out[n + 1, :] = out[n, :] + dt * g(out[n, :], t[n])

    return out, t


# Høyreside for ODE: u'(t) = A u - F
def g(u, t):
    return A @ u - F


# Initialbetingelse (f.eks. u(x,0) = 0 på indre punkter)
u0 = 1.0 + X[1:-1] + 5 * np.sin(np.pi * X[1:-1])

#   Kjører forlengs euler
u_varme, t = euler(g, u0, 0, 10, 10000)  # Må bruke mange steg for å få et stabilt svar

print("dt=", t[1] - t[0])

x_inner = X[1:-1]

# Plot noen tidsskiver
plt.figure(figsize=(6, 4))
plt.plot(X[1:-1], u_varme[1, :], label="t1 ≈ {:.3f}".format(t[100]))
plt.plot(X[1:-1], u_varme[10, :], label="t2 ≈ {:.3f}".format(t[500]))
plt.plot(X[1:-1], u_varme[-1, :], label="t3 ≈ {:.3f}".format(t[-1]))
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Varmeligning: $u_t = u_{xx} - f(x)$ med $f(x)=\\cos(\\pi x)$")
plt.legend()
plt.grid(True)
plt.show()

# %%
