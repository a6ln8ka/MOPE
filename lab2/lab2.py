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
        result[i] = result[i]/5
    return result


def dete(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
           - a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
           + a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]))


variant = 111
m = 5
y_max = (30 - variant) * 10
y_min = (20 - variant) * 10
x1_min = 10
x1_max = 60
x2_min = -70
x2_max = -10

# y1 = []
# y2 = []
# y3 = []
# y4 = []
# y5 = []
# for i in range(4):
#     y1.append(random.randint(y_min, y_max))
#     y2.append(random.randint(y_min, y_max))
#     y3.append(random.randint(y_min, y_max))
#     y4.append(random.randint(y_min, y_max))
#     y5.append(random.randint(y_min, y_max))

x_n = [[-1, 1, -1], [-1, -1, 1]]

y1 = [9, 15, 20]
y2 = [10, 14, 18]
y3 = [11, 10, 12]
y4 = [15, 12, 10]
y5 = [9, 14, 16]

print("матриця планування:\n X1 | X2 | Y1 | Y2 | Y3 | Y4 | Y5")
for i in range(3):
    print(f"{x_n[0][i]:^3} |{x_n[1][i]:^4}|{y1[i]:^4}|{y2[i]:^4}|{y3[i]:^4}|{y4[i]:^4}|{y5[i]:^4}")

avg_y = get_avg([y1, y2, y3, y4, y5])
print("Середнє значення функції відгуку в рядку: y\u03041 = {:.3f},"
      " y\u03042 = {:.3f},"
      " y\u03043 = {:.3f}".format(avg_y[0], avg_y[1], avg_y[2]))

sigma = get_dispersion([y1, y2, y3, y4, y5], avg_y)
print("Значення дисперсії по рядках: \u03c3\u00b2(y1) = {:.2f},"
      " \u03c3\u00b2(y2) = {:.2f},"
      " \u03c3\u00b2(y3) = {:.2f}".format(sigma[0], sigma[1], sigma[2]))

major_deviation = math.sqrt((2 * (2 * m - 2)) / (m * (m - 4)))
print("Основне відхилення: ", round(major_deviation, 3))

Fuv1 = sigma[0] / sigma[1]
Fuv2 = sigma[2] / sigma[0]
Fuv3 = sigma[2] / sigma[1]
print("\nF\u1d64\u1d651 = {:.4f}\nF\u1d64\u1d652 = {:.4f}\nF\u1d64\u1d653 = {:.4f}".format(Fuv1, Fuv2, Fuv3))

Ouv1 = ((m - 2) / m) * Fuv1
Ouv2 = ((m - 2) / m) * Fuv2
Ouv3 = ((m - 2) / m) * Fuv3
print("\n\u03b8\u1d64\u1d651 = {:.4f}\n"
      "\u03b8\u1d64\u1d652 = {:.4f}\n"
      "\u03b8\u1d64\u1d653 = {:.4f}".format(Ouv1, Ouv2, Ouv3))

Ruv1 = abs(Ouv1 - 1)/major_deviation
Ruv2 = abs(Ouv2 - 1)/major_deviation
Ruv3 = abs(Ouv3 - 1)/major_deviation
print("\nR\u1d64\u1d651 = {:.4f}\nR\u1d64\u1d652 = {:.4f}\nR\u1d64\u1d653 = {:.4f}".format(Ruv1, Ruv2, Ruv3))

mx1 = sum(x_n[0])/3
mx2 = sum(x_n[1])/3
my = sum(avg_y)/3
print("\nРозрахунок нормованих коефіцієнтів рівняння регресії:\n"
      "mx1 = {:.3f}\n"
      "mx2 = {:.3f}\n"
      "my = {:.3f}".format(mx1, mx2, my))

a1 = (x_n[0][0]*x_n[0][0] + x_n[0][1]*x_n[0][1] + x_n[0][2]*x_n[0][2])/3
a2 = (x_n[0][0]*x_n[1][0] + x_n[0][1]*x_n[1][1] + x_n[0][2]*x_n[1][2])/3
a3 = (x_n[1][0]*x_n[1][0] + x_n[1][1]*x_n[1][1] + x_n[1][2]*x_n[1][2])/3
print("a1 = {:.3f}, a2 = {:.3f}, a3 = {:.3f}".format(a1, a2, a3))

a11 = (x_n[0][0] * avg_y[0] + x_n[0][1] * avg_y[1] + x_n[0][2] * avg_y[2])/3
a22 = (x_n[1][0] * avg_y[0] + x_n[1][1] * avg_y[1] + x_n[1][2] * avg_y[2])/3
print("a11 = {:.3f}, a22 = {:.3f}".format(a11, a22))

b0 = dete([[my, mx1, mx2], [a11, a1, a2], [a22, a2, a3]]) / dete([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]])
b1 = dete([[1, my, mx2], [mx1, a11, a2], [mx2, a22, a3]]) / dete([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]])
b2 = dete([[1, mx1, my], [mx1, a1, a11], [mx2, a2, a22]]) / dete([[1, mx1, mx2], [mx1, a1, a2], [mx2, a2, a3]])
print("b0 = {:.3f}, b1 = {:.3f}, b2 = {:.3f}".format(b0, b1, b2))
print("Отже нормоване рівняння регресії y = {:.3f} + {:.3f}*x1 + {:.3f}*x2".format(b0, b1, b2))

print("Натуралізація коефіцієнтів")
delta_x1 = abs(x1_max - x1_min) / 2
delta_x2 = abs(x2_max - x2_min) / 2
x10 = (x1_max + x1_min) / 2
x20 = (x2_max + x2_min) / 2
print("\u0394x1 = {},"
      "\u0394x2 = {},"
      "x10 = {},"
      "x20 = {},".format(delta_x1, delta_x2, x10, x20))

a0 = b0 - b1 * (x10/delta_x1) - b2 * (x20/delta_x2)
a1 = b1 / delta_x1
a2 = b2 / delta_x2
print("a0 = {:.3f}, a1 = {:.3f}, a2 = {:.3f}".format(a0, a1, a2))
print("Натуралізоване рівняння регресії y = {:.3f} + {:.3f}*x1 + {:.3f}*x2".format(a0, a1, a2))

