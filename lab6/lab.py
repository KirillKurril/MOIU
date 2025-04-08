import numpy as np

def quadratic_programming(c, D, A, b, x_init, J_b, J_b_star, max_iter=100):
    x = np.array(x_init, dtype=float)
    n = len(c)
    
    for _ in range(max_iter):
        c_x = c + D @ x
        A_b = A[:, list(J_b)]
        A_b_inv = np.linalg.inv(A_b)
        u_x = -c_x[list(J_b)].T @ A_b_inv
        Delta_x = u_x @ A + c_x.T
        
        if np.all(Delta_x >= -1e-10):
            return x
        
        j0 = np.argmin(Delta_x)
        l = np.zeros(n)
        l[j0] = 1
        
        D_star = D[np.ix_(list(J_b_star), list(J_b_star))]
        A_b_star = A[:, list(J_b_star)]
        H_top = np.hstack((D_star, A_b_star.T))
        H_bottom = np.hstack((A_b_star, np.zeros((A_b_star.shape[0], A_b_star.shape[0]))))
        H = np.vstack((H_top, H_bottom))
        
        b_star = np.concatenate((D[:, j0][list(J_b_star)], A[:, j0]))
        x_H = -np.linalg.inv(H) @ b_star
        l[list(J_b_star)] = x_H[:len(J_b_star)]
        
        delta = l.T @ D @ l
        if delta <= 1e-10:
            return None
        
        theta_j0 = abs(Delta_x[j0]) / delta
        
        thetas = []
        for j in J_b:
            if l[j] < 0:
                thetas.append(-x[j]/l[j])
            else:
                thetas.append(np.inf)
        thetas.append(theta_j0)
        
        theta0 = min(thetas)
        if theta0 == np.inf:
            return None
        
        min_idx = thetas.index(theta0)
        if min_idx < len(J_b):
            j_star = list(J_b)[min_idx]
        else:
            j_star = j0
        
        x = x + theta0 * l
        
        if j_star == j0:
            J_b_star = J_b_star.union({j0})
        elif j_star in J_b:
            if j_star in J_b_star:
                J_b_star = J_b_star - {j_star}
        else:
            s = list(J_b).index(j_star)
            A_b_inv_aj = A_b_inv @ A[:, j_star]
            if not np.allclose(A_b_inv_aj[s], 0):
                J_b = J_b - {j_star} | {j0}
                J_b_star = J_b_star - {j_star} | {j0}
            else:
                J_b = J_b - {j_star} | {j0}
                J_b_star = J_b_star - {j_star} | {j0}
    
    return x

c = np.array([-8, -6, -4, -6])
D = np.array([[2, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
A = np.array([[1, 0, 2, 1], [0, 1, -1, 2]])
b = np.array([2, 3])
x_init = [2, 3, 0, 0]
J_b = {0, 1}
J_b_star = {0, 1}

result = quadratic_programming(c, D, A, b, x_init, J_b, J_b_star)
print(result)