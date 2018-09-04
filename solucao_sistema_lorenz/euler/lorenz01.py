import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros

N = 3000 
M = 3
vx = zeros((M, N))
vy = zeros((M, N))
vz = zeros((M, N))

x0 = np.array([5.000, 5.005, 5.05])
y0 = np.array([5.000, 5.005, 5.05])
z0 = np.array([5.000, 5.005, 5.05])

# legendas
x1 = np.array([5.000, 5.005, 5.05])
y1 = np.array([5.000, 5.005, 5.05])
z1 = np.array([5.000, 5.005, 5.05])

dt = 0.005
a = 10
b = 8/3
r = 490/17

def lorenz(x, y, z, a, b, r):
    for k in range(N):
        for j in range(3):
            x[j] = x[j] + dt*a*(y[j] - x[j])
            y[j] = y[j] + dt*(r*x[j] - y[j] - x[j]*z[j])
            z[j] = z[j] + dt*(x[j]*y[j] - b*z[j])
            vx[j][k] = x[j]
            vy[j][k] = y[j]
            vz[j][k] = z[j]
    plt.subplot(3,1,1)
    for i in range(len(vx)):
        plt.plot(vx[i], label=f"x0 = {x1[i]}")
    plt.grid(linestyle='-.')
    plt.xlabel('x')
    plt.title('Equacoes de Lorenz')
    #plt.legend(loc='lower left')
    plt.subplot(3,1,2)
    for i in range(len(vy)):
        plt.plot(vy[i], label=f"y0 = {y1[i]}")
    plt.grid(linestyle='-.')
    plt.xlabel('y')
    #plt.legend(loc='lower left')
    plt.subplot(3,1,3)
    for i in range(len(vz)):
        plt.plot(vz[i], label=f"z0 = {z1[i]}")
    plt.grid(linestyle='-.')
    plt.xlabel('z')
    #plt.legend(loc='lower left')
    plt.show()


lorenz(x0, y0, z0, a, b, r)
