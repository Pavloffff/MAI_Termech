import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

def sector(x, y, r):
    cx = [x + r * np.cos(i / 100) for i in range(314, 628)]
    cy = [y + r * np.sin(i / 100) for i in range(314, 628)]
    return (cx, cy)

def circle(x, y, r):
    cx = [x + r * np.cos(i / 100) for i in range(0, 628)]
    cy = [y + r * np.sin(i / 100) for i in range(0, 628)]
    return (cx, cy)

t = sp.Symbol('t')
R = 1
r = 0.2
m1 = 5
m2 = 3.5
l = 0.65
g = 9.80665
phi = sp.sin(np.pi / 6 * t)
dphi = sp.diff(phi, t)
ddphi = sp.diff(dphi, t)
theta = sp.sin(np.pi / 4 * t)
dtheta = sp.diff(theta, t)
ddtheta = sp.diff(dtheta, t)

x2 = (R - r) * sp.sin(ddtheta)
y2 = (-1) * (R - r) * sp.cos(ddtheta)
vx2 = sp.diff(x2, t)
vy2 = sp.diff(y2, t)
ax2 = sp.diff(vx2, t)
ay2 = sp.diff(vy2, t)
x3 = x2 + l * sp.sin(ddphi * sp.cos(phi-theta) + dphi ** 2 * sp.sin(phi-theta))
y3 = y2 - l * sp.cos(ddphi * sp.cos(phi-theta) + dphi ** 2 * sp.sin(phi-theta))
vx3 = sp.diff(x3, t)
vy3 = sp.diff(y3, t)

T = np.linspace(0, 50, 1000)
X1 = np.zeros_like(T)
Y1 = np.zeros_like(T)
X2 = np.zeros_like(T)
Y2 = np.zeros_like(T)
X3 = np.zeros_like(T)
Y3 = np.zeros_like(T)
V = np.zeros_like(T)

x = np.linspace(0, 0, 1000)
y = np.linspace(-1.05, 0, 1000)

for i in np.arange(len(T)):
    X2[i] = sp.Subs(x2, t, T[i])
    Y2[i] = sp.Subs(y2, t, T[i])
    X3[i] = sp.Subs(x3, t, T[i])
    Y3[i] = sp.Subs(y3, t, T[i])

fig = plt.figure(figsize=[16, 9])
ax = fig.add_subplot(1, 2, 2)
ax.axis('equal')
ax.set(xlim=[-1.5, 1.5], ylim=[-3, 1.5])
ax.plot(x, y, linestyle = '--', linewidth = 1, color = 'black')
line1, = ax.plot([X2[0], X2[0] + l * sp.sin(np.pi)], [Y2[0], Y2[0] + l * sp.cos(np.pi)], 'black')
line2, = ax.plot([X1[0], X2[0]], [Y1[0], Y2[0]], linestyle = '--', linewidth = 1, color = 'black')
line3, = ax.plot([X2[0], X2[0]], [Y2[0], y[0]], linestyle = '--', linewidth = 1, color = 'black')
sector1, = ax.plot(sector(X1[0], Y1[0], R)[0], sector(X1[0], Y1[0], R)[1], 'red')
circle1, = ax.plot(circle(X2[0], Y2[0], r)[0], circle(X2[0], Y2[0], r)[1], 'blue')
circle2, = ax.plot(circle(X3[0], Y3[0], 0.1)[0], circle(X3[0], Y3[0], 0.1)[1], 'green')
point, = ax.plot(0, 0, marker='o', color='black')
point1, = ax.plot(X2[0], Y2[0], marker='o', color='black')
point2, = ax.plot(X3[0], Y3[0], marker='o', color='black')

def kadr(i):
    point.set_data(0, 0)
    point1.set_data(X2[i], Y2[i])
    point2.set_data(X3[i], Y3[i])
    line1.set_data([X2[i], X3[i]], [Y2[i], Y3[i]])
    line2.set_data([X1[i], X2[i]], [Y1[i], Y2[i]])
    line3.set_data([X2[i], X2[i]], [Y2[i], 0 - R])
    sector1.set_data(sector(X1[i], Y1[i], R)[0], sector(X1[i], Y1[i], R)[1])
    circle1.set_data(circle(X2[i], Y2[i], r)[0], circle(X2[i], Y2[i], r)[1])
    circle2.set_data(circle(X3[i], Y3[i], 0.1)[0], circle(X3[i], Y3[i], 0.1)[1])
    return sector1, circle1, circle2, point, point1, point2, line1, line2, line3

ax2 = fig.add_subplot(4, 2, 1)
ax2.plot(T, X2)
plt.title('Vx[O1]')
plt.xlabel('t')
plt.ylabel('Vx')

ax3 = fig.add_subplot(4, 2, 3)
ax3.plot(T, Y2)
plt.title('Vy[01]')
plt.xlabel('t')
plt.ylabel('Vy')

ax4 = fig.add_subplot(4, 2, 5)
ax4.plot(T, X3)
plt.title('Vx[A]')
plt.xlabel('t')
plt.ylabel('Vx')

ax5 = fig.add_subplot(4, 2, 7)
ax5.plot(T, Y3)
plt.title('Vy[A]')
plt.xlabel('t')
plt.ylabel('Vy')

model = FuncAnimation(fig,
                      kadr, 
                      interval=(T[1] - T[0]) * 1000,
                      frames=len(T))

plt.show()
