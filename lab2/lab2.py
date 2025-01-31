import numpy as np

def get_input():
    choice = input("Выберите режим (1 - использовать предзаготовленные данные, 2 - ввести вручную): ")
    if choice == '1':
        return predefined_data()
    elif choice == '2':
        return manual_input()
    else:
        print("Некорректный ввод. Повторите попытку.")
        return get_input()

def predefined_data():
    c = np.array([1, 1, 0, 0, 0], dtype=float)
    A = np.array([[-1, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 1]], dtype=float)
    x = np.array([0, 0, 1, 3, 2], dtype=float)
    B = np.array([2, 3, 4])  
    return c, A, x, B

def manual_input():
    try:
        m = int(input("Введите количество ограничений (m): "))
        n = int(input("Введите количество переменных (n): "))
        
        print("Введите коэффициенты целевой функции c (разделенные пробелом): ")
        c = np.array(list(map(float, input().split())), dtype=float)
        
        print(f"Введите матрицу коэффициентов A ({m}x{n}):")
        A = np.array([list(map(float, input().split())) for _ in range(m)], dtype=float)
        
        print("Введите начальный базисный план x (разделенные пробелом): ")
        x = np.array(list(map(float, input().split())), dtype=float)
        
        print("Введите базисные индексы (разделенные пробелом): ")
        B = np.array(list(map(int, input().split()))) - 1
        
        return c, A, x, B
    except ValueError:
        print("Ошибка ввода. Повторите попытку.")
        return manual_input()

def simplex_method(c, A, x, B):
    m, n = A.shape
    print("Шаг 1: Формируем базисную матрицу AB")
    while True:
        AB = A[:, B]
        print(f"Базисная матрица AB:\n{AB}")
        
        AB_inv = np.linalg.inv(AB)
        print(f"Обратная матрица AB_inv:\n{AB_inv}")
        
        cB = c[B]
        print(f"Шаг 2: Вектор cB: {cB}")
        
        u = cB @ AB_inv
        print(f"Шаг 3: Вектор потенциалов u: {u}")
        
        delta = u @ A - c
        print(f"Шаг 4: Вектор оценок Δ: {delta}")
        
        if np.all(delta >= 0):
            print(f"Шаг 5: Оптимальный план найден: {x}")
            return x
        
        j0 = np.where(delta < 0)[0][0]
        print(f"Шаг 6: Выбран индекс j0 = {j0}")
        
        z = AB_inv @ A[:, j0]
        print(f"Шаг 7: Вектор z: {z}")
        
        theta = np.where(z > 0, x[B] / z, np.inf)
        print(f"Шаг 8: Вектор θ: {theta}")
        
        theta0 = np.min(theta)
        print(f"Шаг 9: Минимальное значение θ0: {theta0}")

        if np.isinf(theta0):
            print("Шаг 10: Целевая функция не ограничена сверху.")
            return None
        
        k = np.where(theta == theta0)[0][0]
        j_star = B[k]
        print(f"Шаг 11: Индекс j* = {j_star}")
        
        B[k] = j0
        print(f"Шаг 12: Новый базис B: {B}")
        
        x[B] = x[B] - theta0 * z  
        x[j0] = theta0
        x[j_star] = 0
        print(f"Шаг 13: Новый план x: {x}")

if __name__ == "__main__":
    c, A, x, B = get_input()
    result = simplex_method(c, A, x, B)
    if result is not None:
        print("Оптимальный план:", result)
    else:
        print("Целевая функция не ограничена сверху.")
