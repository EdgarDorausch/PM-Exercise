from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

f = lambda t, y: -2*y
y_0 = 0.5
t_min = 0.0
t_max = 2.0

y_ana = lambda t: 0.5*np.exp(-2*t)
dt = 0.1


def explicit_euler(f: Callable[[float, float],float], t_start, t_stop, dt, y_start):
    t = np.arange(start=t_start, stop=t_stop, step=dt)
    y = np.empty(t.shape+y_start.shape)

    y[0] = y_start

    for i in range(0, t.shape[0]-1):

        y[i+1] = y[i] + dt*f(t[i],y[i])

    return t, y

t_num, y_num = explicit_euler(f, 0.0, 2.0, 0.3, np.array([0.5]))
plt.plot(t_num, y_num)

t_ana = np.linspace(t_min, t_max)
plt.plot(t_ana, y_ana(t_ana))

plt.show()