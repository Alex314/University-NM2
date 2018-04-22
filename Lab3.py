import numpy as np
from algs import thomas


c = [1, 0.1, 0.001]

b_arr = np.array([[2.63, 2.88, 3.21, 3.5, 3.75, 3.96, 4.13, 4.26, 4.35, 7.4],
                  [0.155, -0.144, -0.381, -0.676, -1.029, -1.44, -1.909, -2.436, -3.021, -0.664],
                  [-0.11725, -0.47664, -0.77601, -1.13536, -1.55469,
                   -2.034, -2.57329, -3.17256, -3.83181, -1.55104]])

up = np.array([0.2 + i * 0.2 for i in range(9)])
center = np.array([-5.5 - 0.1 * i for i in range(10)])
down = np.array([0.4 + 0.1 * i for i in range(9)])

ans = np.array([-0.5, -0.6, -0.7, -0.8, -0.9, -1, -1.1, -1.2, -1.3, -1.4])

diagonal = [ci * center for ci in c]
x = [thomas(up, diagonal[i], down, b_arr[i]) for i in range(3)]

for i in range(3):
    print("c = ", c[i])
    dif = np.abs(x[i] - ans)
    print('Abs diff:', dif)
    print('Mean:', dif.mean(), end='\n\n')
