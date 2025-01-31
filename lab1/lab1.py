import numpy as np

def is_inverse_correct(A, A_inv):
    identity = np.eye(len(A))
    return np.allclose(np.dot(A, A_inv), identity)

def update_inverse(A, A_inv, x, i):
    try:
        print("\nШаг 1: Находим ℓ = A^(-1) * x")
        l = np.dot(A_inv, x)
        print("ℓ =", l)
        
        if np.isclose(l[i], 0):
            print("Матрица A' необратима.")
            return None  
        
        print("\nШаг 2: Заменяем i-й элемент ℓ на -1")
        l_ = l.copy()
        l_[i] = -1
        print("Измененный вектор ℓ* =", l)
        
        print("\nШаг 3: Находим 𝑙^ = -1/ℓ[i] * ℓ*")
        l_hat = -l_ / l[i]
        print("𝑙̂ =", l_hat)
        
        print("\nШаг 4: Формируем матрицу Q")
        Q = np.eye(len(A))
        Q[:, i] = l_hat
        print("Q =\n", Q)
        
        print("\nШаг 5: Вычисляем A'^{-1} = Q * A^{-1}")
        A_new_inv = np.dot(Q, A_inv)
        print("Новая обратная матрица A'^{-1} =\n", A_new_inv)
        
        return A_new_inv
    except Exception as e:
        print("Ошибка при вычислении новой обратной матрицы:", e)
        return None

def safe_input_matrix(n, prompt):
    while True:
        try:
            print(prompt)
            matrix = np.array([list(map(float, input().split())) for _ in range(n)])
            if matrix.shape != (n, n):
                raise ValueError("Матрица должна быть квадратной.")
            return matrix
        except ValueError as e:
            print("Ошибка ввода:", e, "Попробуйте снова.")

def safe_input_vector(n, prompt):
    while True:
        try:
            print(prompt)
            vector = np.array(list(map(float, input().split())))
            if vector.shape != (n,):
                raise ValueError("Вектор должен содержать ровно {} элементов.".format(n))
            return vector
        except ValueError as e:
            print("Ошибка ввода:", e, "Попробуйте снова.")

def safe_input_index(n, prompt):
    while True:
        try:
            index = int(input(prompt))
            if not (0 <= index < n):
                raise ValueError("Индекс должен быть от 0 до {}".format(n - 1))
            return index
        except ValueError as e:
            print("Ошибка ввода:", e, "Попробуйте снова.")

n = int(input("Введите размерность матрицы: "))
A = safe_input_matrix(n, "Введите матрицу A:")
A_inv = safe_input_matrix(n, "Введите обратную матрицу A^{-1}:")

if not is_inverse_correct(A, A_inv):
    print("Введенная матрица A^{-1} не является обратной для A. Проверьте данные.")
else:
    x = safe_input_vector(n, "Введите вектор x:")
    i = safe_input_index(n, "Введите индекс заменяемого столбца (начиная с 0): ")
    
    update_inverse(A, A_inv, x, i)
