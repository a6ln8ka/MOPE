import random
import numpy as np
import math
from _decimal import Decimal
from functools import reduce
from itertools import compress
import scipy.stats
from scipy.stats import f, t


def f(x1, x2, x3):
    coef = [6.7, 7.1, 7.8, 7.4, 8.2, 0.2, 9.5, 6.6, 0.1, 0.7, 7.1]
    return regression_equation(x1, x2, x3, coef)


def add_sq_nums(x):
    for i in range(len(x)):
        x[i][3] = x[i][0] * x[i][1]
        x[i][4] = x[i][0] * x[i][2]
        x[i][5] = x[i][1] * x[i][2]
        x[i][6] = x[i][0] * x[i][1] * x[i][2]
        x[i][7] = x[i][0] ** 2
        x[i][8] = x[i][1] ** 2
        x[i][9] = x[i][2] ** 2
    return x


def plan_matrix5(x_norm):
    l = 1.73
    x_norm = np.array(x_norm)
    x_norm = np.transpose(x_norm)
    x = np.ones(shape=(len(x_norm), len(x_norm[0])))
    for i in range(8):
        for j in range(3):
            if x_norm[i][j] == -1:
                x[i][j] = x_range[j][0]
            else:
                x[i][j] = x_range[j][1]
    for i in range(8, len(x)):
        for j in range(3):
            x[i][j] = float((x_range[j][0] + x_range[j][1]) / 2)
    dx = [x_range[i][1] - (x_range[i][0] + x_range[i][1]) / 2 for i in range(3)]
    x[8][0] = (-l * dx[0]) + x[9][0]
    x[9][0] = (l * dx[0]) + x[9][0]
    x[10][1] = (-l * dx[1]) + x[9][1]
    x[11][1] = (l * dx[1]) + x[9][1]
    x[12][2] = (-l * dx[2]) + x[9][2]
    x[13][2] = (l * dx[2]) + x[9][2]
    x = add_sq_nums(x)
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = round(x[i][j], 3)
    return x.tolist()


def generate_factors_table(raw_array):
    raw_list = [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]] + list(map(lambda x: x ** 2, row)) for row in raw_array]
    return list(map(lambda row: list(map(lambda el: round(el, 3), row)), raw_list))


def generate_y(m, factors_table):
    return [[round(f(row[0], row[1], row[2]) + random.randint(-5, 5), 3) for _ in range(m)] for row in factors_table]


def cochran_criteria(m, N, y_table):
    def get_cochran_value(f1, f2, q):
        partResult1 = q / f2
        params = [partResult1, f1, (f2 - 1) * f1]
        fisher = scipy.stats.f.isf(*params)
        result = fisher / (fisher + (f2 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()
    y_variations = [np.var(i) for i in y_table]
    max_y_variation = max(y_variations)
    gp = max_y_variation/sum(y_variations)
    f1 = m - 1
    f2 = N
    p = 0.95
    q = 1-p
    gt = get_cochran_value(f1, f2, q)
    print("Gp = {}, Gt = {}".format(gp, gt))
    if gp < gt:
        print("Дисперсія однорідна")
        return True
    else:
        print("Дисперсія неоднорідна")
        return False


def set_factors_table(factors_table):
    def x_i(i):
        with_null_factor = list(map(lambda x: [1] + x, generate_factors_table(factors_table)))
        res = [row[i] for row in with_null_factor]
        return np.array(res)
    return x_i


def m_ij(*arrays):
    return np.average(reduce(lambda accum, el: accum*el, list(map(lambda el: np.array(el), arrays))))


def find_coefficients(factors, y_vals):
    x_i = set_factors_table(factors)
    coefficients = [[m_ij(x_i(column), x_i(row)) for column in range(11)] for row in range(11)]
    y_numpy = list(map(lambda row: np.average(row), y_vals))
    free_values = [m_ij(y_numpy, x_i(i)) for i in range(11)]
    beta_coefficients = np.linalg.solve(coefficients, free_values)
    return list(beta_coefficients)


def print_equation(coefficients, importance=[True]*11):
    x_i_names = list(compress(["", "*x1", "*x2", "*x3", "*x12", "*x13", "*x23", "*x123", "*x1^2", "*x2^2", "*x3^2"], importance))
    coefficients_to_print = list(compress(coefficients, importance))
    equation = " ".join(["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), coefficients_to_print)), x_i_names)])
    print("Рівняння регресії: y = " + equation)


def student_criteria(m, N, y_table, beta_coefficients):
    def get_student_value(f3, q):
        return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001')).__float__()
    average_variation = np.average(list(map(np.var, y_table)))
    x_i = set_factors_table(natural_plan)
    variation_beta_s = average_variation/N/m
    standard_deviation_beta_s = math.sqrt(variation_beta_s)
    t_i = np.array([abs(beta_coefficients[i])/standard_deviation_beta_s for i in range(len(beta_coefficients))])
    f3 = (m-1)*N
    q = 0.05
    t_our = get_student_value(f3, q)
    importance = [1 if el > t_our else 0 for el in list(t_i)]
    d = sum(importance)
    # print result data
    print("βs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), beta_coefficients))))
    print("ts: " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
    print("f3 = {}; q = {}; tтабл = {}".format(f3, q, t_our))
    print("d =", d)
    print_equation(beta_coefficients, importance)
    return importance


def regression_equation(x1, x2, x3, coef, importance = [True]*11):
    factors_array = [1, x1, x2, x3, x1*x2, x1*x3, x2*x3, x1*x2*x3, x1**2, x2**2, x3**2]
    return sum([el[0]*el[1] for el in compress(zip(coef, factors_array), importance)])


def fisher_criteria(m, N, d, x_table, y_table, b_coefficients, importance):
    def get_fisher_value(f3, f4, q):
        return Decimal(abs(scipy.stats.f.isf(q, f4, f3))).quantize(Decimal('.0001')).__float__()
    f3 = (m - 1) * N
    f4 = N - d
    q = 0.05
    theoretical_y = np.array([regression_equation(row[0], row[1], row[2], b_coefficients) for row in x_table])
    # print(theoretical_y)
    average_y = np.array(list(map(lambda el: np.average(el), y_table)))
    s_ad = m/(N-d) * sum((theoretical_y - average_y)**2)
    # print(s_ad)
    y_variations = np.array(list(map(np.var, y_table)))
    s_v = np.average(y_variations)
    f_p = float(s_ad/s_v)
    f_t = get_fisher_value(f3, f4, q)
    theoretical_values_to_print = list(zip(map(lambda x: "x1 = {0[1]:<10} x2 = {0[2]:<10} x3 = {0[3]:<10}".format(x), x_table), theoretical_y))
    print("Fp = {}, Ft = {}".format(f_p, f_t))
    print("Рівняння регресії адекватно оригіналу при рівні значимості 0.05" if f_p < f_t else "Рівняння регресії неадекватно оригіналу при рівні значимості 0.05")
    return True if f_p < f_t else False


def main(disp_amount, x_norm):
    x_nat = plan_matrix5(x_norm)
    m = 3
    N = 14
    x_norm = np.transpose(np.array(x_norm))
    global natural_plan
    natural_plan = generate_factors_table(x_nat)
    y_arr = generate_y(m, x_nat)
    if disp_amount == 0:
        print("Матриця планування:\n  X1  |  X2  |  X3  |X1*X2|X1*X3|X2*X3|X1*X2*X3|  X1\u00b2 |  X2\u00b2 |  X3\u00b2")
        for i in range(14):
            print(
                f"{x_norm[i][0]:^6}|{x_norm[i][1]:^6}|{x_norm[i][2]:^6}|{x_norm[i][3]:^5}|{x_norm[i][4]:^5}|{x_norm[i][5]:^5}|{x_norm[i][6]:^8}|{x_norm[i][7]:^6}|{x_norm[i][8]:^6}|{x_norm[i][9]:^6}")

        print("Матриця планування з натуралізованими значеннями факторів:"
              "\n  X1  |  X2  |  X3  | X1*X2 | X1*X3 | X2*X3 | X1*X2*X3 |   X1\u00b2  |   X2\u00b2  |  X3\u00b2")
        for i in range(14):
            print(
                f"{x_nat[i][0]:^6}|{x_nat[i][1]:^6}|{x_nat[i][2]:^6}|{x_nat[i][3]:^7}|{x_nat[i][4]:^7}|{x_nat[i][5]:^7}|{x_nat[i][6]:^10}|{x_nat[i][7]:^8}|{x_nat[i][8]:^8}|{x_nat[i][9]:^6}")

        print("     Y1     |     Y2     |     Y3")
        for i in range(14):
            print("{:^12}|{:^12}|{:^12}".format(y_arr[i][0], y_arr[i][1], y_arr[i][2]))
    while not cochran_criteria(m, N, y_arr):
        m += 1
    if cochran_criteria(m, N, y_arr):
        disp_amount += 1
    y_arr = generate_y(m, natural_plan)
    coefficients = find_coefficients(natural_plan, y_arr)
    print_equation(coefficients)
    importance = student_criteria(m, N, y_arr, coefficients)
    d = len(list(filter(None, importance)))
    fisher_criteria(m, N, d, natural_plan, y_arr, coefficients, importance)

    return disp_amount


l = 1.73
x1min = 10
x1max = 60
x2min = -70
x2max = -10
x3min = 60
x3max = 70
x_norm = [[-1, -1, -1, -1, 1, 1, 1, 1, -1.73, 1.73, 0, 0, 0, 0],
          [-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -1.73, 1.73, 0, 0],
          [-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -1.73, 1.73],
          [1, 1, -1, -1, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, -1, 1, -1, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0],
          [1, -1, -1, 1, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0],
          [-1, 1, 1, -1, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 2.9929, 2.9929, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2.9929, 2.9929, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2.9929, 2.9929]]

x_range = [[x1min, x1max], [x2min, x2max], [x3min, x3max]]


disp_amount = 0
i = 0
while disp_amount < 10:
    disp_amount = main(disp_amount, x_norm)
    i += 1

print("Додаткове завзання")
print("Кількість ітерацій", i)
print("Кількість однорідних дисперцій", disp_amount)
