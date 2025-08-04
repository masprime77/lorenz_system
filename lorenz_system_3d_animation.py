import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# Parámetros de Lorenz
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# Definir ecuaciones del sistema de Lorenz
def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Tiempo y condiciones iniciales
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
initial_state = [1.0, 1.0, 1.0]

# Resolver el sistema
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
x, y, z = sol.y

# Crear marcos para la animación
frames = []
step = 20  # cada cuántos puntos añadir una nueva línea
for i in range(step, len(x), step):
    frames.append(go.Frame(data=[go.Scatter3d(
        x=x[:i], y=y[:i], z=z[:i],
        mode='lines',
        line=dict(color='royalblue', width=3)
    )]))

# Figura inicial vacía
fig = go.Figure(
    data=[go.Scatter3d(
        x=x[:step], y=y[:step], z=z[:step],
        mode='lines',
        line=dict(color='royalblue', width=3)
    )],
    layout=go.Layout(
        title="Lorenz Atractor 3D (Interactivo + Animado)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=0., y=2.5, z=0.5)
            )
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None])]
        )]
    ),
    frames=frames
)

# Exportar a HTML interactivo
fig.write_html("lorenz_3d_animation.html")

# También mostrarlo en Jupyter si estás usando uno
fig.show()