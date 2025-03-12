import numpy as np

def dual_simplex(A, c, b, B):
    m, n = A.shape
    B = np.array(B) - 1
    N = np.array([j for j in range(n) if j not in B])  

    while True:
        AB_inv = np.linalg.inv(A[:, B])

        cB = c[B]

        y = cB @ AB_inv

        kB = AB_inv @ b

        k = np.zeros(n)
        k[B] = kB

        if np.all(k >= 0):
            return k

        jk_index = np.argmin(kB)

        delta_y = AB_inv[jk_index, :]

        mu = {j: delta_y @ A[:, j] for j in N}

        if all(mu_j >= 0 for mu_j in mu.values()):
            return "Задача несовместна"

        sigma = {j: (c[j] - A[:, j] @ y) / mu[j] for j in N if mu[j] < 0}

        j0 = min(sigma, key=sigma.get)

        B[jk_index] = j0
        N = np.array([j for j in range(n) if j not in B])

c = np.array([-4, -3, -7, 0, 0])  
A = np.array([[-2, -1, -4, 1, 0], 
              [-2, -2, -2, 0, 1]])
b = np.array([-1, -1.5])  
B = [4, 5]  

result = dual_simplex(A, c, b, B)
print("Оптимальный план:", result)
