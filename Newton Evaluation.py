import numpy as np

def hermite_divided_differences(x_nodes, y_nodes, dy_nodes):
    n = len(x_nodes)
    m = 2 * n
    z = np.zeros(m)
    table = np.zeros((m, m))

    # Duplicate nodes and initialize column 0
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

    # Columns 2 to m-1: standard recursive divided differences
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