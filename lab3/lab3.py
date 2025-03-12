import numpy as np
from scipy.optimize import linprog

c = np.array([1, 0, 0])
A = np.array([[1, 1, 1], [2, 2, 2]])  
b = np.array([0, 0])  

m, n = A.shape
A_aux = np.hstack([A, np.eye(m)])
c_aux = np.concatenate([np.zeros(n), -np.ones(m)])

print("Вспомогательная задача:")
print(f"Матрица A:\n{A_aux}")
print(f"Вектор c: {c_aux}")
print(f"Вектор b: {b}")

res_aux = linprog(c_aux, A_eq=A_aux, b_eq=b, method='highs')

if res_aux.success and np.all(res_aux.x[n:] == 0):
    B = np.where(res_aux.x[:n] > 0)[0] + 1  
    x = res_aux.x[:n] 
    print(f"Допустимый план: {x}")
    print(f"Базисные индексы: {B}")
else:
    print("Задача несовместна")