import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros

N = 3000

vx = zeros(N)
vy = zeros(N)
vz = zeros(N)

x0 = 5.0
y0 = 5.0
z0 = 5.0

x1 = 5.005
y1 = 5.005
z1 = 5.005

dt = 0.005
a = 10
b = 8/3
r = 470/19


def lorenz(x, y, z, a, b, r):
    for k in range(N):
        x = x + dt*a*(y - x)
        y = y + dt*(r*x - y - x*z)
        z = z + dt*(x*y - b*z)
        vx[k] = x
        vy[k] = y
        vz[k] = z

    plt.subplot(3, 1, 1)
    plt.plot(vx, vy)
    plt.grid(linestyle='-.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Equacoes de Lorenz')

    plt.subplot(3, 1, 2)
    plt.plot(vx, vz)
    plt.grid(linestyle='-.')
    plt.xlabel('x')
    plt.ylabel('z')

    plt.subplot(3, 1, 3)
    plt.plot(vy, vz)
    plt.grid(linestyle='-.')
    plt.xlabel('y')
    plt.ylabel('z')
    #plt.show()


lorenz(x0, y0, z0, a, b, r)
lorenz(x1, y1, z1, a, b, r)
plt.show()
