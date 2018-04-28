import numpy as np
import matplotlib.pyplot as plt


def calculate(interval, n, p, g, k, inner_cond, outer_cond, t_interval, dt, T0, alpha):
    r1, r2 = interval
    d = (r2 - r1) / n
    t1, t2 = t_interval
    nt = int((t2 - t1) / dt)
    r = np.linspace(r1, r2, n+1)
    t = np.linspace(t1, t2, nt+1)
    T = np.zeros((len(t), len(r)))
    T[0] = np.ones_like(r) * T0
    beta = alpha * dt / d**2

    for i in range(1, len(T)):
        T[i, 1:-1] = T[i-1, :-2] * beta + T[i-1, 1:-1] * (1 - 2 * beta)\
                        + T[i-1, 2:] * beta + beta * g * d**2 / k
        # Inner border conditions
        if inner_cond[0] == 2 and p == 0:
            qa = inner_cond[1]
            T[i, 0] = (T[i - 1, 1] + 2 * d * qa / k) * beta + \
                      T[i - 1, 0] * (1 - 2 * beta) + T[i - 1, 1] * beta + beta * g * d ** 2 / k
        else:
            raise ValueError
        # Outer border conditions
        if outer_cond[0] == 3 and p == 0:
            h, t_inf = outer_cond[1:]
            T[i, -1] = T[i - 1, -2] * beta + \
                       T[i - 1, -1] * (1 - 2 * beta) + \
                       ((t_inf - T[i - 1, -1]) * 2 * d * h / k + T[i - 1, -2]) * beta + \
                       beta * g * d ** 2 / k
        else:
            raise ValueError
    return T


t_arr = [0, 20, 40, 60, 80, 100, 200]
t_interval = (0, 200)
dt = 0.01

indexes = [int(t / dt) for t in t_arr]

n = 60
p = 0  # Coordinate system
r1, r2 = 0.0, 0.6
g = 2E5
k = 50
alpha = 4e-3
t_0 = 80
# Inner border conditions
qa = -5E4
i_cond = (2, qa)
# Outer border conditions
h = 400
t_inf = 80
o_cond = (3, h, t_inf)

T = calculate(interval=(r1, r2), n=n, p=p, g=g, k=k, inner_cond=i_cond, outer_cond=o_cond,
              t_interval=t_interval, dt=dt, T0=t_0, alpha=alpha)

for i in indexes:
    plt.plot(np.linspace(r1, r2, n+1), T[i], label='t = ' + str(int(i*dt)) + ' c')
plt.grid()
plt.legend()
plt.xlabel('$r$, м')
plt.ylabel('$T$, °C')
plt.show()
