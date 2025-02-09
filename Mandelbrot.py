import numpy as np
import matplotlib.pyplot as plt


def compute_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter):
    """Compute iteration counts for the Mandelbrot set in the specified region."""
    a = np.linspace(x_min, x_max, width)
    b = np.linspace(y_min, y_max, height)
    A, B = np.meshgrid(a, b)

    iter_counts = np.full(A.shape, max_iter, dtype=np.int32)
    x, y = A.copy(), B.copy()
    mask = np.ones_like(A, dtype=bool)

    for step in range(max_iter):
        x_prev, y_prev = x.copy(), y.copy()
        x = x_prev ** 2 - y_prev ** 2 - A
        y = 2 * x_prev * y_prev - B

        mag_sq = x ** 2 + y ** 2
        escaped = (mag_sq > 4) & mask
        iter_counts[escaped] = step
        mask[escaped] = False

        if not mask.any():
            break

    return iter_counts


def generate_image(iter_counts, max_iter, x_min, x_max, y_min, y_max):
    """Generate and display the Mandelbrot image from iteration counts."""
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max_iter)
    image = cmap(norm(iter_counts))

    # Set black color for points that reached max_iter
    image[iter_counts == max_iter] = [0, 0, 0, 1]

    plt.imshow(image, origin='lower', extent=[x_min, x_max, y_min, y_max])
    plt.axis('off')
    plt.show()


def main():
    # Configuration parameters
    width, height = 800, 800
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    max_iter = 100
    use_second_pass = True  # Set to False to disable smoothing

    # First pass computation
    first_pass = compute_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter)

    if use_second_pass:
        # Calculate second pass with offset grid
        dx = (x_max - x_min) / width
        dy = (y_max - y_min) / height
        second_pass = compute_mandelbrot(
            x_min + dx / 2, x_max - dx / 2,
            y_min + dy / 2, y_max - dy / 2,
            width, height, max_iter
        )
        # Average the results
        final_counts = (first_pass + second_pass) // 2
    else:
        final_counts = first_pass

    generate_image(final_counts, max_iter, x_min, x_max, y_min, y_max)


if __name__ == "__main__":
    main()