def lagrange_custom(x_nodes, y_nodes, x):
    n = len(x_nodes)
    result = 0.0

    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if j != i:
                L_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * L_i

    return result