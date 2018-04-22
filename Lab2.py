import numpy as np
import math
import matplotlib.pyplot as plt
from algs import rk3 as rk4, adams_bashford_5


def get_yd(x, y_list):
    return np.array([df(x, y_list[0])])


def yd2(x, y_list):
    """y''=sin x

    k' = sin x
    y' = k

    y0' = sin x
    y1' = y0
    """
    return np.array([math.sin(x),
                     y_list[0]])


f = lambda x: -2 * math.cos(x) ** 2 + math.cos(x)
df = lambda x, y: math.sin(2 * x) - y * math.tan(x)
f_np = np.vectorize(f)
y_0 = [-1]
h = [0.1 * 5 ** -i for i in range(3)]
interval = (0, 10)
print('Steps:', h)

plt.subplot(3, 2, 1)
plt.ylabel('Answer')

for hi in h:
    ans = adams_bashford_5(interval[0], interval[1], step=hi, y0_list=y_0, f=get_yd)
    xp = ans[:, 0]
    yp = ans[:, 1]
    plt.subplot(3, 2, 1)
    if abs(hi - 0.02) < 1e-10:
        hi = 0.02
    plt.plot(xp, yp, label='h=' + str(hi))
    if hi is h[-1]:
        plt.subplot(3, 1, 3)
        plt.plot(xp, yp, label='Best AB5')
    plt.subplot(3, 2, 3)
    plt.grid(True)
    e = yp - f_np(xp)
    plt.plot(xp, e, label='h=' + str(hi))

plt.ylabel('Absolute error')

for hi in h:
    ans = rk4(interval[0], interval[1], step=hi, y0_list=y_0, f=get_yd)
    xp = ans[:, 0]
    yp = ans[:, 1]
    plt.subplot(3, 2, 2)
    if abs(hi - 0.02) < 1e-10:
        hi = 0.02
    plt.plot(xp, yp, label='h=' + str(hi))
    if hi is h[-1]:
        plt.subplot(3, 1, 3)
        plt.plot(xp, yp, label='Best RK3')
    plt.subplot(3, 2, 4)
    plt.grid(True)
    e = yp - f_np(xp)
    plt.plot(xp, e, label='h=' + str(hi))


dotnum = 4000
xplot = [interval[0] + i*(interval[1] - interval[0]) / dotnum for i in range(dotnum + 1)]
yplot = [f(x) for x in xplot]

plt.subplot(3, 2, 1)
plt.grid(True)
plt.title('Adams-Bushford 5')
plt.plot(xplot, yplot, label='True answer')
plt.legend()
plt.subplot(3, 2, 2)
plt.grid(True)
plt.title('Runge–Kutta 3')
plt.plot(xplot, yplot, label='True answer')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(xplot, yplot, label='True answer', color='r')

plt.grid(True)
plt.legend()
plt.show()
