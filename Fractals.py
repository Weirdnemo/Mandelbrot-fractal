import numpy as np
import matplotlib.pyplot as plt

# Parameters
width, height = 800, 800
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
max_iter = 100
c = -0.7 + 0.27015j  # Initial value of c

# Create a grid of complex numbers
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Function to compute the Julia set
def julia_set(Z, c, max_iter):
    output = np.zeros(Z.shape, dtype=int)
    for i in range(max_iter):
        mask = np.abs(Z) < 1000  # Escape radius
        Z[mask] = Z[mask] ** 2 + c
        output[mask] = i
    return output

# Create a custom colormap: black background, white and grey fractal
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 0), (1, 1, 1), (0.75, 0.75, 0.75)]  # Black -> White -> Grey
cmap = LinearSegmentedColormap.from_list("custom", colors, N=max_iter)

# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(julia_set(Z, c, max_iter), cmap=cmap, extent=(x_min, x_max, y_min, y_max))
ax.set_title("Julia Set Animation")
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")

# Animation loop
frame = 0
try:
    while True:
        # Change the parameter c to create movement
        c = 0.7885 * np.exp(1j * frame * 0.02)
        im.set_data(julia_set(Z, c, max_iter))
        ax.set_title(f"Julia Set with c = {c:.4f}")
        fig.canvas.draw()
        fig.canvas.flush_events()  # Update the plot
        frame += 1
        plt.pause(0.05)  # Add a small delay to control the speed
except KeyboardInterrupt:
    print("Animation stopped.")