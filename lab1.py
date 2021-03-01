import random


def gen_x(n=8, minimum=0, maximum=20):
    x = []
    for i in range(n):
        x.append(random.randint(minimum, maximum))
    return x


def gen_x0(x1, x2, x3):
    x0 = [(min(x1) + max(x1)) / 2, (min(x2) + max(x2)) / 2, (min(x3) + max(x3)) / 2]
    return x0


def gen_dx(x1, x2, x3, x0):
    dx = [max(x1) - x0[0], max(x2) - x0[1], max(x3) - x0[2]]
    return dx


def gen_y(a, x1, x2, x3):
    y = []
    for i in range(8):
        y.append(a[0] + a[1] * x1[i] + a[2] * x2[i] + a[3] * x3[i])
    return y


def gen_xh(x, x0, dx):
    xh = (x-x0)/dx
    return xh


a0 = 9
a1 = 12
a2 = 4
a3 = 6
a = [a0, a1, a2, a3]

x1 = [11, 14, 17, 13, 1, 16, 13, 2]
x2 = [13, 15, 19, 18, 7, 7, 16, 13]
x3 = [9, 12, 5, 16, 11, 17, 18, 11]

# calculate x0, dx, y
x0 = gen_x0(x1, x2, x3)
dx = gen_dx(x1, x2, x3, x0)
y = gen_y(a, x1, x2, x3)

# calculate normalized value
xh = [[], [], []]
for i in range(8):
    xh[0].append(gen_xh(x1[i], x0[0], dx[0]))
    xh[1].append(gen_xh(x2[i], x0[1], dx[1]))
    xh[2].append(gen_xh(x3[i], x0[2], dx[2]))

# ptint values
print("N   X1   X2   X3    Y      XH1   XH2   XH3")
for i in range(8):
    print(f"{i+1:^1} |{x1[i]:^4} {x2[i]:^4} {x3[i]:^4}|"
          f"{y[i]:^5} || {'%.2f' %xh[0][i]:^5} {'%.2f' %xh[1][i]:^5} {'%.2f'%xh[2][i]:^5} |")

print("x0 =", x0)
print("dx =", dx)

# calculate Yет
yet = a[0] + a[1] * x0[0] + a[2] * x0[1] + a[3] * x0[2]
print("Yет =", yet)

# find optimal value
# ymax = max(y)
# for i in range(8):
#     if y[i] == ymax:
#         print("max(Y) = {} = Y({}, {}, {})".format(ymax, x1[i], x2[i], x3[i]))
#         break

# variant 121
avg_y = sum(y)/len(y)
sorted_y = sorted(y)
for i in range(len(y)-1):
    if sorted_y[i] < avg_y < sorted_y[i + 1]:
        optimal_y = sorted_y[i]
        print(optimal_y)
        for j in range(len(y)):
            if y[j] == optimal_y:
                print("Optimal value: {} = Y({}, {}, {})".format(optimal_y, x1[j], x2[j], x3[j]))
                break
