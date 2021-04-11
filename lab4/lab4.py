import random
import math


def get_avg(array):
    result = []
    for i in range(len(array[0])):
        result.append(0)
        for j in range(len(array)):
            result[i] += array[j][i]
        result[i] = result[i]/len(array)
    return result


def get_dispersion(array, avg_y):
    result = []
    for i in range(len(array[0])):
        result.append(0)
        for j in range(len(array)):
            result[i] += (array[j][i] - avg_y[i])**2
        result[i] = result[i]/3
    return result


x = [[-25, -5], [-30, 45], [-5, 5]]
m = 3
n = 8

avg_x_min = sum(x[i][0] for i in range(3))/3
avg_x_max = sum(x[i][1] for i in range(3))/3
yi_max = 200 + avg_x_max
yi_min = 200 + avg_x_min
x_norm = [[1, 1, 1, 1, 1, 1, 1, 1],
          [-1, -1, 1, 1, -1, -1, 1, 1],
          [-1, 1, -1, 1, -1, 1, -1, 1],
          [1, 1, -1, -1, -1, -1, 1, 1]]

x_matr = [[1, 1, 1, 1, 1, 1, 1, 1],
          [-25, -25, -5, -5, -25, -25, -5, -5],
          [-30, 45, -30, 45, -35, 45, -30, 45],
          [5, 5, -5, -5, -5, -5, 5, 5]]

f1 = m - 1
f2 = n
f3 = f1 * f2
q = 0.05

y = []
for i in range(m):
    y.append([])
    for j in range(n):
        y[i].append(random.randint(yi_min, yi_max))

x12 = []
x13 = []
x23 = []
x123 = []
for i in range(len(x_matr[0])):
    x12.append(x_matr[1][i] * x_matr[2][i])
    x13.append(x_matr[1][i] * x_matr[3][i])
    x23.append(x_matr[2][i] * x_matr[3][i])
    x123.append(x_matr[1][i] * x_matr[2][i] * x_matr[3][i])

print("Матриця планування:\n X0 | X1 | X2 | X3 | X12 | X13 | X23 | X123 | Y1 | Y2 | Y3")
for i in range(8):
    print(f"{x_matr[0][i]:^4}|{x_matr[1][i]:^4}|{x_matr[2][i]:^4}|{x_matr[3][i]:^4}|"
          f"{x12[i]:^5}|{x13[i]:^5}|{x23[i]:^5}|{x123[i]:^6}|"
          f"{y[0][i]:^4}|{y[1][i]:^4}|{y[2][i]:^4}")
print("Нормована матриця:\n X0 | X1 | X2 | X3")
for i in range(8):
    print(f"{x_norm[0][i]:^4}|{x_norm[1][i]:^4}|{x_norm[2][i]:^4}|{x_norm[3][i]:^4}|")

avg_y = get_avg(y)
print("Середнє значення функції відгуку в рядку:\ny\u03041 = {:.3f},"
      " y\u03042 = {:.3f},"
      " y\u03043 = {:.3f},"
      " y\u03044 = {:.3f},"
      " y\u03045 = {:.3f},"
      " y\u03046 = {:.3f},"
      " y\u03047 = {:.3f},"
      " y\u03048 = {:.3f},".format(avg_y[0], avg_y[1], avg_y[2], avg_y[3], avg_y[4], avg_y[5], avg_y[6], avg_y[7]))

mx1 = sum(x_matr[1]) / n
mx2 = sum(x_matr[2]) / n
mx3 = sum(x_matr[3]) / n
my = sum(avg_y) / n
print("mx1 = {:.3f}, mx2 = {:.3f}, mx3 = {:.3f}, my = {:.3f}".format(mx1, mx2, mx3, my))


y_sum = sum(avg_y)
b = [0] * 8
b[0] = y_sum
for i in range(8):
    b[1] += avg_y[i] * x_norm[0][i]
    b[2] += avg_y[i] * x_norm[1][i]
    b[3] += avg_y[i] * x_norm[2][i]
    b[4] += avg_y[i] * x_norm[0][i] * x_norm[1][i]
    b[5] += avg_y[i] * x_norm[0][i] * x_norm[2][i]
    b[6] += avg_y[i] * x_norm[1][i] * x_norm[2][i]
    b[7] += avg_y[i] * x_norm[0][i] * x_norm[1][i] * x_norm[2][i]

for i in range(8):
    b[i] = b[i] / n

result = []
for i in range(8):
    result.append(b[0] + b[1] * x_matr[1][i] + b[2] * x_matr[2][i] + b[3] * x_matr[3][i] + b[4] * x12[i] + b[5] * x13[i] + b[6] * x23[i] + b[7] * x123[i])

print("b0 = {:.3f}, b1 = {:.3f}, b2 = {:.3f}, b3 = {:.3f},"
      " b12 = {:.3f}, b13 = {:.3f}, b23 = {:.3f}, b123 = {:.3f}".format(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]))

print("Підставимо значення факторів з матриці планування")
for i in range(8):
    print("{:.3f} + x1 * {:.3f}"
          " + x2 * {:.3f} + x3 * {:.3f}"
          " + x12 * {:.3f} + x13 * {:.3f}"
          " + x23 * {:.3f} + x123 * {:.3f}"
          " = {:.3f}".format(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], result[i]))

sigma = get_dispersion(y, avg_y)
print("Значення дисперсії по рядках: \u03c3\u00b2(y1) = {:.2f},"
      " \u03c3\u00b2(y2) = {:.2f},"
      " \u03c3\u00b2(y3) = {:.2f},"
      " \u03c3\u00b2(y4) = {:.2f},"
      " \u03c3\u00b2(y5) = {:.2f},"
      " \u03c3\u00b2(y6) = {:.2f},"
      " \u03c3\u00b2(y7) = {:.2f},"
      " \u03c3\u00b2(y8) = {:.2f}".format(sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], sigma[5], sigma[6], sigma[7]))

gp = max(sigma)/sum(sigma)
print("Gp = ", gp)
f1 = m-1
f2 = n
if gp < 0.5157:
    print("Дисперсія однорідна")
else:
    print("Дисперсія неоднорідна")
    exit()

sb = sum(sigma) / n
s2bs = sb / (n * m)
sbs = math.sqrt(s2bs)
print("S\u00b2b = {:.3f}, S\u00b2(\u03b2s) = {:.3f}, S(\u03b2s) = {:.3f}".format(sb, s2bs, sbs))

betha = []
for i in range(4):
    betha.append((avg_y[0] * x_norm[i][0] + avg_y[1] * x_norm[i][1] + avg_y[2] * x_norm[i][2] + avg_y[3] * x_norm[i][3]) / 4)
print("\u03b20 = {:.3f}, \u03b21 = {:.3f}, \u03b22 = {:.3f}, \u03b23 = {:.3f}".format(betha[0], betha[1], betha[2], betha[3]))

t = []
for i in range(4):
    t.append(betha[i]/sbs)
print("t0 = {:.3f}, t1 = {:.3f},"
      " t2 = {:.3f}, t3 = {:.3f},".format(t[0], t[1], t[2], t[3]))

f3 = f1 * f2
t_table = 2.101

y_ = [0, 0, 0, 0, 0, 0, 0, 0]
d = 0
nm = []
for i in range(4):
    if t[i] < t_table:
        nm.append(i)
        d += 1
    else:
        for j in range(4):
            if i == 0:
                y_[j] += b[i]
            else:
                y_[j] += b[i] * x_matr[2][j]

print("d = {}".format(d))
print("Підставимо значення факторів з матриці планування, y =", y_[0], y_[1], y_[2], y_[3])

s2ad = (m / (n - d)) * sum((y_[i] - avg_y[i])**2 for i in range(4))
print("S\u00b2ад = {:.3f}".format(s2ad))

Fp = s2ad / sb
f4 = n - d
print("Критерій Фішера Fp =", Fp)

F8_table = [5.3, 4.5, 4.1, 3.8, 3.7, 3.6, 3.3, 3.1, 2.9]
if Fp < F8_table[int(f4-1)]:
    print("Рівняння регресії адекватно оригіналу при рівні значимості 0.05")
else:
    print("Рівняння регресії неадекватно оригіналу при рівні значимості 0.05")
