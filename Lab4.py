import numpy as np
from algs import thomas
import matplotlib.pyplot as plt


def decompose(interval, n, p, g, k, inner_cond, outer_cond):
    r1, r2 = interval
    d = (r2 - r1) / n
    r = np.linspace(r1, r2, n+1)
    up = (1 + p * d / 2 / r)[:-1]
    mid = -2 * np.ones(n+1)
    down = (1 - p * d / 2 / r)[1:]
    b = -g * d**2 / k * np.ones(n+1)
    # Inner border conditions
    if inner_cond[0] == 3:
        h, t_inf = inner_cond[1:]
        mid[0] = -2 - 2 * d * h / k * (1 - p * d / 2 / r1)
        up[0] = 2
        b[0] = -g * d**2 / k - 2 * h * d * t_inf / k * (1 - p * d / 2 / r1)
    else:
        raise ValueError
    # Outer border conditions
    if outer_cond[0] == 1:
        t_b = outer_cond[1]
        down[-1] = 0
        mid[-1] = 1
        b[-1] = t_b
    else:
        raise ValueError
    return up, mid, down, b


n = 50
p = 2  # Coordinate system
r1, r2 = 0.2, 0.4
g = 3.6E6
k = 80
# Inner border conditions
h = 500
t_0 = 50
# Outer border conditions
t_b = 50

u, c, d, b = decompose(interval=(r1, r2), n=n, p=p, g=g, k=k, inner_cond=(3, h, t_0), outer_cond=(1, t_b))
t = thomas(u, c, d, b)

qa = -k * (-3 * t[0] + 4 * t[1] - t[2]) / 2 / (r2 - r1) * n
qa_3 = -k * (-11 * t[0] + 18 * t[1] - 9 * t[2] + 2 * t[3]) / 6 / (r2 - r1) * n
qb = k * (3 * t[-1] - 4 * t[-2] + t[-3]) / 2 / (r2 - r1) * n
print('Qa:', int(qa), int(qa_3), int(h * (t_0 - t[0])))
print('Qb:', int(qb))

plt.plot(np.linspace(r1, r2, n+1), t)
plt.grid()
plt.xlabel('$r$, м')
plt.ylabel('$T$, °C')
plt.show()
