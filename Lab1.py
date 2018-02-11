import numpy as np
import math
import matplotlib.pyplot as plt


def get_yd(x, y_list):
    return np.array([math.sin(2*x) - y_list[0] * math.tan(x)])


def RK3(x0, x1, step, y0_list, f=get_yd):
    assert x1 > x0
    n = round((x1 - x0) / step)
    n = 1 if n < 1 else n
    h = (x1 - x0) / n
    print('Wanted step:', step)
    print('Finaly step is:', h)
    print('Num of steps:', n)
    yk = y0_list
    for i in range(n):
        xk = x0 + i * h
        k1 = h * f(xk, yk)
        k2 = h * f(xk + h / 3, yk + k1 / 3)
        k3 = h * f(xk + 2 * h / 3, yk + 2 * k2 / 3)
        yk = yk + (k1 + 3 * k3) / 4
        yield xk, yk


f = lambda x: -2 * math.cos(x) ** 2 + math.cos(x)
df = lambda x, y: math.sin(2*x) - y * math.tan(x)
f_np = np.vectorize(f)
y_0 = [-1]
h = [0.5 * 5 ** -i for i in range(3)]
print('Step:', h)

interval = (0, 10)
for hi in h:
    ans = [dot for dot in RK3(interval[0], interval[1], step=hi, y0_list=y_0)]
    xp = np.array([i[0] for i in ans])
    yp = np.array([i[1][0] for i in ans])
    plt.subplot(2, 1, 1)
    plt.plot(xp, yp, label='h=' + str(hi))
    plt.subplot(2, 1, 2)
    e = yp - f_np(xp)
    plt.plot(xp, e, label='h=' + str(hi))


dotnum = 4000

xplot = [interval[0] + i*(interval[1] - interval[0]) / dotnum for i in range(dotnum + 1)]
yplot = [f(x) for x in xplot]

plt.title('Error')
plt.legend()
plt.subplot(2, 1, 1)
plt.title('f(x)')
plt.plot(xplot, yplot, label='True answer')
plt.legend()
plt.show()
