import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Target Function, Derivative & Hermite Routines
# -------------------------------------------------------------
def f(x):
    return (2.0 * x) / (1.0 + x**2)

def df(x):
    return (2.0 * (1.0 - x**2)) / ((1.0 + x**2)**2)

def hermite_divided_differences(x_nodes, y_nodes, dy_nodes):
    n = len(x_nodes)
    m = 2 * n  # 2 * 6 = 12 entries
    z = np.zeros(m)
    table = np.zeros((m, m))

    # Duplicate nodes: z = [x0, x0, x1, x1, ...]
    for i in range(n):
        z[2 * i] = x_nodes[i]
        z[2 * i + 1] = x_nodes[i]
        table[2 * i, 0] = y_nodes[i]
        table[2 * i + 1, 0] = y_nodes[i]

    # Column 1: derivatives on even rows, divided differences on odd rows
    for row in range(m - 1):
        if row % 2 == 0:
            table[row, 1] = dy_nodes[row // 2]
        else:
            table[row, 1] = (table[row + 1, 0] - table[row, 0]) / (z[row + 1] - z[row])

    # Columns 2 to m-1: standard divided differences
    for col in range(2, m):
        for row in range(m - col):
            numerator = table[row + 1, col - 1] - table[row, col - 1]
            denominator = z[row + col] - z[row]
            table[row, col] = numerator / denominator

    return z, table[0, :]

def hermite_eval(z_nodes, coeffs, x):
    m = len(coeffs)
    result = coeffs[0]
    term = 1.0
    for i in range(1, m):
        term *= (x - z_nodes[i - 1])
        result += coeffs[i] * term
    return result

# -------------------------------------------------------------
# 2. Nodes Setup: 6 Nodes on [-1.5, 1.5] (Degree 11 Polynomial)
# -------------------------------------------------------------
x_nodes = np.linspace(-1.5, 1.5, 6)
y_nodes = f(x_nodes)
dy_nodes = df(x_nodes)

z_nodes, coeffs = hermite_divided_differences(x_nodes, y_nodes, dy_nodes)

# -------------------------------------------------------------
# 3. Dense Grid Evaluation
# -------------------------------------------------------------
x_dense = np.linspace(-1.5, 1.5, 400)
y_true = f(x_dense)
y_hermite = [hermite_eval(z_nodes, coeffs, x) for x in x_dense]

# Calculate max absolute error across the grid
max_error = np.max(np.abs(y_true - y_hermite))
print(f"Maximum Error on [-1.5, 1.5]: {max_error:.2e}")

# -------------------------------------------------------------
# 4. Plotting
# -------------------------------------------------------------
plt.figure(figsize=(9, 6))

# Original function curve
plt.plot(x_dense, y_true, 'k--', linewidth=2.2, label=r"Original $f(x) = \frac{2x}{1+x^2}$")

# Hermite polynomial curve (Degree 11)
plt.plot(x_dense, y_hermite, color='#8E44AD', linewidth=2.0, label=r"Hermite $H_{11}(x)$ (6 nodes)")

# Interpolation nodes
plt.scatter(x_nodes, y_nodes, color='red', s=60, zorder=5, label="6 Nodes")

# Plot styling
plt.title("Comparison: Original Function vs. Hermite Interpolation (6 Nodes)", fontsize=13, fontweight='bold')
plt.xlabel("x", fontsize=11)
plt.ylabel("y", fontsize=11)
plt.ylim(-1.3, 1.3)
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc="upper left", fontsize=10)

plt.savefig("hermite_6_nodes_clean_plot.png", dpi=300, bbox_inches='tight')
print("Plot successfully saved as hermite_6_nodes_clean_plot.png")
plt.show()