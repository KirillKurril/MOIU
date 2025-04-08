import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(matrix, cost, u=None, v=None, title="Transport Plan", basis=None):
    m, n = matrix.shape
    labels = np.array([[f"{int(matrix[i,j])}/{int(cost[i,j])}" if (basis and (i,j) in basis) else f"-/{int(cost[i,j])}" for j in range(n)] for i in range(m)])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(matrix, annot=labels, fmt="s", cmap="YlGnBu", cbar=False,
                linewidths=0.5, linecolor='gray', xticklabels=False, yticklabels=False, ax=ax)

    if u is not None:
        ax.set_yticks(np.arange(m) + 0.5)
        ax.set_yticklabels([f"u={round(val, 1)}" for val in u], rotation=0)

    if v is not None:
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels([f"v={round(val, 1)}" for val in v], rotation=90)

    plt.title(title)
    plt.xlabel("Demand (potentials v)")
    plt.ylabel("Supply (potentials u)")
    plt.tight_layout()
    plt.show(block=False)


def place_marks_on_B(B_marks, basis):
    current_sign = not B_marks[basis]
    for (i, j) in B_marks.keys():
        if basis[0] == i or basis[1] == j:
            if B_marks[(i, j)] is None:
                B_marks[(i, j)] = current_sign
                place_marks_on_B(B_marks, (i, j))


def potentials_method(supply, demand, cost):
    total_supply = np.sum(supply)
    total_demand = np.sum(demand)
    if total_supply > total_demand:
        demand = np.append(demand, total_supply - total_demand)
        cost = np.hstack((cost, np.zeros((len(supply), 1))))
    elif total_supply < total_demand:
        supply = np.append(supply, total_demand - total_supply)
        cost = np.vstack((cost, np.zeros((1, len(demand)))))


    m = len(supply)
    n = len(demand)
    x = np.zeros((m, n))
    basis = []
    i = 0
    j = 0
    supply_left = supply.copy()
    demand_left = demand.copy()

    while i < m and j < n:
        allocation = min(supply_left[i], demand_left[j])
        x[i, j] = allocation
        basis.append((i, j))
        supply_left[i] -= allocation
        demand_left[j] -= allocation
        if i == m - 1 and j == n - 1:
            break
        if np.isclose(supply_left[i], 0) and i < m - 1:
            i += 1
        elif np.isclose(demand_left[j], 0) and j < n - 1:
            j += 1
        else:
            if i < m - 1:
                i += 1
            if j < n - 1:
                j += 1

    iteration = 0
    while True:
        iteration += 1

        A_eq = []
        b_eq = []
        for (i_cell, j_cell) in basis:
            row = [0] * (m + n)
            row[i_cell] = 1
            row[m + j_cell] = 1
            A_eq.append(row)
            b_eq.append(cost[i_cell, j_cell])
        eq_fix = [0] * (m + n)
        eq_fix[0] = 1
        A_eq.append(eq_fix)
        b_eq.append(0)

        A_eq = np.array(A_eq, dtype=float)
        b_eq = np.array(b_eq, dtype=float)

        potentials = np.linalg.solve(A_eq, b_eq)
        u = potentials[:m]
        v = potentials[m:]

        plot_heatmap(x, cost, u, v, title=f"Transport Plan Iteration {iteration}", basis=basis)

        candidate = None
        for i_cell in range(m):
            for j_cell in range(n):
                if (i_cell, j_cell) not in basis and (u[i_cell] + v[j_cell] - cost[i_cell, j_cell]) > 1e-10:
                    candidate = (i_cell, j_cell)
                    break
            if candidate is not None:
                break

        if candidate is None:
            plot_heatmap(x, cost, u, v, title="Final Optimal Transport Plan", basis=basis)
            return

        basis.append(candidate)
        basis.sort()
        cycle = deepcopy(basis)
        for i_cell in range(m):
            count = sum(1 for (p, q) in cycle if p == i_cell)
            if count <= 1:
                cycle = [cell for cell in cycle if cell[0] != i_cell]
        for j_cell in range(n):
            count = sum(1 for (p, q) in cycle if q == j_cell)
            if count <= 1:
                cycle = [cell for cell in cycle if cell[1] != j_cell]

        cycle_marks = {cell: None for cell in cycle}
        cycle_marks[candidate] = True
        place_marks_on_B(cycle_marks, candidate)

        theta = np.inf
        for cell, sign in cycle_marks.items():
            if sign is False:
                i_cell, j_cell = cell
                if x[i_cell, j_cell] < theta:
                    theta = x[i_cell, j_cell]

        for cell, sign in cycle_marks.items():
            i_cell, j_cell = cell
            if sign:
                x[i_cell, j_cell] += theta
            else:
                x[i_cell, j_cell] -= theta

        for cell in basis:
            i_cell, j_cell = cell
            if np.isclose(x[i_cell, j_cell], 0) and cell != candidate:
                basis.remove(cell)
                break




def main():
    supply = np.array([100, 300, 300], dtype=float)
    demand = np.array([300, 200, 200], dtype=float)
    cost = np.array([
        [8, 4, 1],
        [8, 4, 3],
        [9, 7, 5]
    ], dtype=float)

    potentials_method(supply, demand, cost)
    input("Press any key to close...")


if __name__ == '__main__':
    main()
