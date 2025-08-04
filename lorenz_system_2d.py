import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parámetros del sistema de Lorenz
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# Definición del sistema de ecuaciones
def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Condiciones iniciales y tiempo
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Resolver el sistema
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

# Extraer variables
x, y, z = sol.y

# Gráficos 2D
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(x, y, lw=0.5)
axs[0].set_title("Plano X vs Y")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")

axs[1].plot(x, z, lw=0.5)
axs[1].set_title("Plano X vs Z")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Z")

axs[2].plot(y, z, lw=0.5)
axs[2].set_title("Plano Y vs Z")
axs[2].set_xlabel("Y")
axs[2].set_ylabel("Z")

plt.tight_layout()
plt.show()