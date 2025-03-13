import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(c, max_iter=100):
    """Computes the Mandelbrot fractal using NumPy vectorization."""
    z = np.zeros_like(c, dtype=np.complex128)
    mask = np.ones(c.shape, dtype=bool)
    iteration_counts = np.zeros(c.shape, dtype=int)

    for i in range(max_iter):
        z[mask] = z[mask]**2 + c[mask]
        mask = (np.abs(z) < 2) & mask  # Keep only points inside the set
        iteration_counts[mask] = i  # Store the number of iterations before escaping

    return iteration_counts

# Visualization area parameters
x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
width, height = 1000, 1000
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

# Compute the Mandelbrot set
mandelbrot_set = mandelbrot(C, max_iter=300)

# Display with a beautiful colormap
plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot_set, extent=(x_min, x_max, y_min, y_max), cmap="inferno")
plt.colorbar(label="Iterations before escape")
plt.title("Mandelbrot Fractal", fontsize=14)
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()
