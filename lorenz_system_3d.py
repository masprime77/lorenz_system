import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del sistema de Lorenz
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# Definimos el sistema de ecuaciones diferenciales
def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Condiciones iniciales y tiempo de simulación
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 50)  # de t=0 a t=50
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Resolver el sistema
solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

# Extraer x, y, z
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]

# Graficar en 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title("Sistema de Lorenz")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter3d(
    x=x, y=y, z=z,
    mode='lines',
    line=dict(color='blue', width=2)
))

fig.update_layout(
    title='Atractor de Lorenz (interactivo)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

# Guardar como archivo HTML
fig.write_html("lorenz_3d.html")

# También puedes mostrarlo en el notebook:
fig.show()

plt.show()