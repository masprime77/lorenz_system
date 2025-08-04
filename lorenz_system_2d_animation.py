import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Sistema de Lorenz ---
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Simulación
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
x, y, z = sol.y

print(f"x range: {x.min()} to {x.max()}")
print(f"y range: {y.min()} to {y.max()}")
print(f"z range: {z.min()} to {z.max()}")

# --- Crear la figura ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].set_title("Plano X vs Z")
axs[1].set_title("Plano X vs Y")
axs[2].set_title("Plano Y vs Z")
axs[0].set_xlabel("X"); axs[0].set_ylabel("Z")
axs[1].set_xlabel("X"); axs[1].set_ylabel("Y")
axs[2].set_xlabel("Y"); axs[2].set_ylabel("Z")
axs[0].set_xlim(min(x), max(x))
axs[0].set_ylim(min(z), max(z))
axs[1].set_xlim(min(x), max(x))
axs[1].set_ylim(min(y), max(y))
axs[2].set_xlim(min(y), max(y))
axs[2].set_ylim(min(z), max(z))
lines = [axs[0].plot([], [], lw=1.5)[0],
         axs[1].plot([], [], lw=1.5)[0],
         axs[2].plot([], [], lw=1.5)[0]]

# --- Funciones de animación ---
def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    lines[0].set_data(x[:frame], z[:frame])
    lines[1].set_data(x[:frame], y[:frame])
    lines[2].set_data(y[:frame], z[:frame])
    return lines

ani = FuncAnimation(fig, update, frames=range(20, len(x), 2), init_func=init, blit=True, interval=17)

# --- Guardar como GIF ---
ani.save("lorenz_2d_animation_xz.gif", writer=PillowWriter(fps=60))
print("GIF guardado correctamente.")

# --- Guardar como MP4 (requiere ffmpeg instalado) ---
ani.save("lorenz_2d_animation_xz.mp4", writer="ffmpeg", fps=60)
print("MP4 guardado correctamente.")

plt.close()