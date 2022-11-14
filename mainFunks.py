import matplotlib.pyplot as plt
import numpy as np
import time
import random
import math
from scipy.integrate import solve_ivp
import joblib
from random import randint

Nstreams = 12
R_data_path = './Data/results/r_data.txt'
Graphic_data_path = './Data/graphics/graphs6and1/saved_fig.png'
FHN_coords_data_path = './Data/FHW_coords.txt'

# Start params
a = 0.7
b = 0.8
S = 0.45
tau1 = 0.08
tau2 = 3.1
tau3 = 1.15

V_inh = -1.5
V_ex = 1.5

# Solving params
k_systems = 4  # num systems
k = 4  # num equels in one system (one model of neuron)
tMax = 300
G_ex = 0.0
g_ex = []
g_inh = []
G_inh = 0.0
highAccuracy = False
for i in range(0, k_systems):
    g_ex.append([])
    g_inh.append([])
    for j in range(0, k_systems):
        if j == i:
            g_inh[i].append(0.0)
            g_ex[i].append(0.0)
        else:
            g_inh[i].append(G_inh)
            g_ex[i].append(G_ex)

# Params for random Generation IC
z1_IC = 0.01
z2_IC = 0

# For plot
scatter_markers = [4, 5, 6, 7, 8, 9, 10, 11]
plot_styles = ['--', '-.', '--', 'dotted', '-.', '--', 'dotted']
plot_colors = ['blue', 'orange', 'green', 'red', 'indigo', 'm', 'yellow']


################################################### functions ##########################################################


def naguma_systems(t, r):

    g_ex = []
    g_inh = []
    for i in range(0, k_systems):
        g_ex.append([])
        g_inh.append([])
        for j in range(0, k_systems):
            if j == i:
                g_inh[i].append(0.0)
                g_ex[i].append(0.0)
            else:
                g_inh[i].append(G_inh)
                g_ex[i].append(G_ex)

    resArr = []
    for i in range(0, k_systems):

        #
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z1_i = r[i*k + 2]
        # z2_i = r[i*k + 3]

        # append fx
        resArr.append((r[i * k] - r[i * k] ** 3 / 3.0 - r[i * k + 1] - r[i * k + 2] * (r[i * k] - V_inh) - r[
            i * k + 3] * (r[i * k] - V_ex) + S) / tau1)
        # append fy
        resArr.append(r[i * k] - b * r[i * k + 1] + a)

        # sum fz
        sumFz1 = 0.0
        sumFz2 = 0.0
        for n in range(0, k_systems):
            sumFz1 += g_inh[i][n] * np.heaviside(r[k * n], 0.0)
            sumFz2 += g_ex[i][n] * np.heaviside(r[k * n], 0.0)
        # append fz1
        resArr.append((sumFz1 - r[i * k + 2]) / tau2)
        # append fz2
        resArr.append((sumFz2 - r[i * k + 3]) / tau3)

    return resArr


def graphics_4_with_G_inh(ginh_, initialConditions):
    startTime = time.time()

    global G_inh, g_inh
    G_inh = ginh_
    g_inh = []
    for i in range(0, k_systems):
        g_inh.append([])
        for j in range(0, k_systems):
            if j == i:
                g_inh[i].append(0.0)
            else:
                g_inh[i].append(G_inh)
    sol = 0
    if highAccuracy:
        sol = solve_ivp(naguma_systems, [0, tMax], initialConditions, rtol=1e-11, atol=1e-11)
    else:
        sol = solve_ivp(naguma_systems, [0, tMax], initialConditions, rtol=1e-8, atol=1e-8)
    xs = []
    ys = []
    z1s = []
    z2s = []

    ts = sol.t

    for i in range(0, k_systems):
        xs.append(sol.y[i * k])
        ys.append(sol.y[i * k + 1])
        z1s.append(sol.y[i * k + 2])
        z2s.append(sol.y[i * k + 3])

    # k_systems_2 = int(k_systems / 2)
    # fig, ax = plt.subplots(k_systems_2, k_systems_2)
    # for i in range(0, k_systems_2):
    #     ax[0][i].plot(ts, xs[i], label=('eq' + str(i + 1)))
    #     ax[0][i].legend()
    #     ax[1][i].plot(ts, xs[i + k_systems_2], label=('eq' + str(i + k_systems_2 + 1)))
    #     ax[1][i].legend()
    #
    # fig.suptitle('G_inh = ' + str(G_inh))
    # plt.show()
    print('g_inh: ', G_inh, '\t solve time: ', time.time() - startTime)
    return xs, ys, ts, G_inh


def findMaximums(X, t):
    maximums = []
    times_of_maximums = []
    indexes_of_maximums = []
    for i in range(200, len(X) - 1):
        if X[i] > X[i - 1] and X[i] > X[i + 1] and X[i] > 0:
            maximums.append(X[i])
            times_of_maximums.append(t[i])
            indexes_of_maximums.append(i)

    return maximums, times_of_maximums, indexes_of_maximums


def findPeriod(inf):
    max, time, index = inf

    time_difference = []
    index_difference = []
    for i in range(1, len(max)):
        time_difference.append(time[i] - time[i - 1])
        index_difference.append(index[i] - index[i - 1])
    time_period = sum(time_difference) / len(time_difference)
    index_period = int(sum(index_difference) / len(index_difference))
    return time_period, index_period


def findPeriod_i(inf, j):
    max, time, index = inf
    return time[j] - time[j - 1], index[j] - index[j - 1]


# Запаздывание между main (первым) нейроном и other(другим)
# Нужно переделать
def lagBetweenNeurons(main_t, main_i, other_t, other_i):
    # передаются времена максимумов главного нейрона и другого, и индексы этих максимумов в массивах
    diff_t = []
    diff_i = []
    if len(main_t) > 1 and abs(main_t[1] - other_t[0]) > abs(main_t[1] - other_t[1]):
        for i in range(1, len(main_t) - 1):
            diff_t.append(-(main_t[i] - other_t[i]))
            diff_i.append(-(main_i[i] - other_i[i]))
    else:
        for i in range(1, len(main_t) - 1):
            diff_t.append(-(main_t[i] - other_t[i - 1]))
            diff_i.append(-(main_i[i] - other_i[i - 1]))
    delay = sum(diff_t) / len(diff_t)
    delay_i = sum(diff_i) / len(diff_i)
    return delay, delay_i


# Запаздывание между main (первым) нейроном и other(другим) на определенном шаге
def lagBetweenNeurons_2(main_t, main_i, other_t, other_i, period, index=3):
    delay = period * 2
    delay_i = 0
    isFirstTry = True
    index_2 = index
    while True:
        if abs(main_t[index] - other_t[index_2 - 1]) > abs(main_t[index] - other_t[index_2]):
            delay = other_t[index_2] - main_t[index]
            delay_i = other_i[index_2] - main_i[index]
        else:
            delay = other_t[index_2] - main_t[index - 1]
            delay_i = other_i[index_2] - main_i[index - 1]

        if delay > period:
            if isFirstTry:
                index_2 = 1
                isFirstTry = False
            else:
                index_2 += 1
        else:
            break

    return delay, delay_i


# Запаздывание между main (первым) нейроном и other(другим) на определенном шаге
def lagBetweenNeurons_3(main_t, main_i, other_t, other_i, period, index=3):
    delay = 0
    delay_i = 0
    # Можно попробовать считать период на каждом шаге
    # period2 = main_t[index] - main_t[index - 1]
    # print('period: ', period, 'period 2:', period2)
    try:
        if abs(main_t[index] - other_t[index - 1]) > abs(main_t[index] - other_t[index]):
            delay = other_t[index] - main_t[index]
            delay_i = other_i[index] - main_i[index]
        else:
            delay = other_t[index] - main_t[index - 1]
            delay_i = other_i[index] - main_i[index - 1]
    except:
        print('Опять сломалось')

        print(len(main_t), len(other_t))
        print(other_t)
        print(other_t[index], main_t[index])


    return delay, delay_i


# Подсчет параметра порядка
def findOrderParam(period, delays):
    sum_re = 0.0
    sum_im = 0.0
    sum_re2 = 0.0
    sum_im2 = 0.0
    for i in range(0, k_systems - 1):
        in_exp = 2 * math.pi * delays[i] / period
        in_exp2 = 4 * math.pi * delays[i] / period
        sum_re += math.cos(in_exp)
        sum_im += math.sin(in_exp)

        sum_re2 += math.cos(in_exp2)
        sum_im2 += math.sin(in_exp2)

    # print('sum_re2', sum_re2, 'sum_im2', sum_im2)
    sum_re += 1.0
    sum_re2 += 1.0
    sum = math.sqrt(sum_re ** 2 + sum_im ** 2)
    sum2 = math.sqrt(sum_re2 ** 2 + sum_im2 ** 2)
    r2 = sum2 / k_systems
    r = sum / k_systems
    return r, r2


def IC_random_generator(a, b):
    random_var = []
    for i in range(0, 2 * k_systems):
        random_var.append(random.uniform(a, b))
    IC_arr = []

    #print('Random IC:')
    for i in range(0, k_systems):
        IC_arr.append(random_var[i])
        IC_arr.append(random_var[i+1])
        IC_arr.append(z1_IC)
        IC_arr.append(z2_IC)
    return np.array(IC_arr)



# print начальные условия
def showInitialConditions(IC, Name='0'):
    if Name == '0':
        print('Initial conditions')
    else:
        print(Name)

    for i in range(0, k_systems):
        print(str(IC[i*k]) + ', ' + str(IC[i*k+1]) + ', ' + str(IC[i*k+2]) + ', ' + str(IC[i*k+3]) + ',')

# Делает solve с случайными или нет НУ (isRand) и рисует x(t)
def solveAndPlotWithIC(G_inh_, isRand, IC, path_graph_x_start=0, path_graph_x_end=0, doNeedShow=False):
    margins = {  # +++
        "left": 0.030,
        "bottom": 0.060,
        "right": 0.995,
        "top": 0.950
    }
    if (isRand == 1):
        IC = IC_random_generator(-2, 2)

    # Нужно ли делать принт НУ?
    if doNeedShow:
        showInitialConditions(IC)
    xs, ys, ts, G_inh = graphics_4_with_G_inh(G_inh_, IC)

    # Нужно сделать так, чтобы при tMax > 200 рисовалось только последняя часть последовательности
    if (tMax > 200):
        if highAccuracy:
            new_len = 12000
        else:
            new_len = 3000
        short_xs_end = []
        short_ts_end = []
        short_xs_start = []
        short_ts_start = []
        for k in range(0, k_systems):
            short_xs_end.append([])
            short_xs_start.append([])
        for i in range(0, new_len):
            for k in range(0, k_systems):
                short_xs_end[k].append(xs[k][-new_len + i])
                short_xs_start[k].append(xs[k][i])
            short_ts_end.append(ts[-new_len + i])
            short_ts_start.append(ts[i])

        # Осцилограмма на первых точках
        plt.figure(figsize=(15, 5))
        plt.subplots_adjust(**margins)
        for i in range(0, k_systems):
            plt.plot(short_ts_start, short_xs_start[i], label=('eq' + str(i + 1)), linestyle=plot_styles[i])
            plt.legend()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Осцилограмма x(t) на первых ' + str(new_len) + ' точках')
        plt.grid()
        # Если передан путь, сохранить график
        if (path_graph_x_start != 0):
            plt.savefig(path_graph_x_start)
        # Нужно ли показывать график
        if doNeedShow:
            plt.show()
        plt.close()

        # Осцилограмма на последних точках
        plt.figure(figsize=(15, 5))
        plt.subplots_adjust(**margins)
        for i in range(0, k_systems):
            plt.plot(short_ts_end, short_xs_end[i],
                label=('eq' + str(i + 1)), linestyle=plot_styles[i], color=plot_colors[i])
            plt.legend()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Осцилограмма x(t) на последних ' + str(new_len) + ' точках')
        plt.grid()

        # Если передан путь, сохранить график
        if (path_graph_x_end != 0):
            plt.savefig(path_graph_x_end)
        # Нужно ли показывать график
        if doNeedShow:
            plt.show()
        plt.close()

    # Полная осцилограмма
    # plt.figure(figsize=(30, 5))
    # for i in range(0, k_systems):
    #     plt.plot(ts, xs[i], label=('eq' + str(i + 1)), linestyle=plot_styles[i])
    #     plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.title('Осцилограмма x(t)')
    # plt.grid()
    # plt.show()

    return xs, ys, ts, G_inh


def recordICAndR(filename, R, IC, G_inh, size):
    # IC - двумерный массив
    # R - двумерный массив
    f = open(filename, 'w')
    f.write(str(k_systems))
    f.write('\n')
    f.write(str(size))
    f.write('\n')
    for i in range(0, size):
        f.write(str(G_inh[i]))
        f.write('\n')
        f.write(str(len(R[i])))
        f.write('\n')
        for j in range(0, len(IC[i])):
            f.write(str(IC[i][j]))
            f.write('\n')
        # f.writelines(str(IC[i]), '\n')
        for j in range(0, len(R[i])):
            f.write(str(R[i][j]))
            f.write('\n')
    f.close()


def classification(period, delays):
    eps = period / 10.0

    for i in range(0, k_systems - 1):
        if delays[i] < 0:
            delays[i] = period + delays[i]
            i = 0
    # print('new delays:', delays)

    # Сколько элементов синхронизированы с главным
    counter0 = 0
    for i in range(0, k_systems - 1):
        if delays[i] < eps or period - delays[i] < eps:
            counter0 += 1
    # print('counter0: ', counter0)
    if counter0 == 2:
        return '3-1'
    if counter0 == 3:
        return '4'

    # Сколько других элементов синхронизированы друг с другом
    counter1 = 0
    for i in range(0, k_systems - 1):
        for j in range(0, k_systems - 1):
            if (i == j):
                continue
            if abs(delays[i] - delays[j]) < eps:
                counter1 += 1
    # print('counter1: ', counter1)

    if counter1 == 0 and counter0 == 1:
        return '2-1-1'
    if counter1 == 0 and counter0 == 2:
        return '3-1'

    if counter1 == 2 and counter0 == 1:
        return '2-2'

    if counter1 == 6 and counter0 == 0:
        return '3-1'

    if counter0 == 0:
        if counter1 == 1:
            return '2-1-1'
        if counter1 == 2:
            return '2-2'

    counter_last = [0, 0, 0]
    for i in range(0, k_systems - 1):
        if delays[i] > period / 4.0 - eps and delays[i] < period / 4.0 + eps:
            counter_last[0] += 1
        if delays[i] > period / 2.0 - eps and delays[i] < period / 2.0 + eps:
            counter_last[1] += 1
        # print(delays[i], 3.0 * period / 4.0 - eps, 3.0 * period / 4.0 + eps, )
        if delays[i] > 3.0 * period / 4.0 - eps and delays[i] < 3.0 * period / 4.0 + eps:
            counter_last[2] += 1
    if counter_last[0] == 1 and counter_last[1] == 1 and counter_last[2] == 1:
        return '1-1-1-1'

    # counter_last = [0,0,0]
    # for i in range(0, k_systems - 1):
    #     if delays[i] > period / 4.0 - 2.0 * eps and delays[i] < period / 4.0 + 2.0 * eps:
    #         counter_last[0] += 1
    #     if delays[i] > period / 2.0 - 2.0 * eps and delays[i] < period / 2.0 + 2.0 * eps:
    #         counter_last[1] += 1
    #     if 3.0 * delays[i] > period / 4.0 - 2.0 * eps and 3.0 * delays[i] < period / 4.0 + 2.0 * eps:
    #         counter_last[2] += 1
    # #print('c_l2: ', counter_last)
    # if counter_last[0] == 1 and counter_last[1] == 1 and counter_last[2] == 1:
    #     return 'doub 1-1-1-1'

    return 'undef'

def exists(path):
    try:
        file = open(path)
    except IOError as e:
        return False
    else:
        return True

def make_FHN_tr(path):
    # Если такого файла еще нет
    if not exists(path):
        global k_systems, G_inh
        G_inh = 0.02
        # Запоминаем k_systems на всякий случай
        k_systems_temp = k_systems
        k_systems = 1

        IC_1_el = np.array([1., 1., 0.01, 0])
        sol = solve_ivp(naguma_systems, [0, 15], IC_1_el, rtol=1e-11, atol=1e-11)


        # Та же штука для 6 элементов
        # Траектория совсем немного другая, но суть та же
        # IC_6_el = np.array([
        #     1.7909191033523797, 0.3947975055180025, 0.01, 0,
        #     -1.3196057166044584, -0.15643994894012603, 0.01, 0,
        #     1.8357397674399445, 0.19450457790747858, 0.01, 0,
        #     -1.9506185972198538, 0.9232371262393431, 0.01, 0,
        #     1.5533010980534387, 0.014229690959855032, 0.01, 0,
        #     -1.5336019316917493, 0.06607850268758791, 0.01, 0
        # ])
        #
        # k_systems = 6
        # sol = solve_ivp(naguma_systems, [0, 15], IC_6_el, rtol=1e-11, atol=1e-11)
        # plt.plot(sol.y[0], sol.y[1], label='6 elems system')
        # plt.plot(sol_1.y[0], sol_1.y[1], label='1elem system')
        # plt.legend()
        # plt.grid()
        # plt.show()

        xs = sol.y[0]
        ys = sol.y[1]
        ts = sol.t

        short_xs = []
        short_ys = []
        short_ts = []

        x_max_arr, t_max_arr, i_max_arr = findMaximums(xs, ts)
        for index in range(i_max_arr[-2], i_max_arr[-1], 1):
            short_xs.append(xs[index])
            short_ys.append(ys[index])
            short_ts.append(ts[index])

        # Длина полученных массивов
        size = len(short_xs)
        # print(len(short_xs))
        # plt.plot(short_ts, short_xs)
        # plt.grid()
        # plt.show()
        # plt.plot(short_xs, short_ys, alpha=0.5)
        # plt.grid()
        # plt.show()

        with open(path, 'w') as f:
            print(size, file=f)
            for i in range(size):
                print(short_xs[i], short_ys[i], file=f)

        # Возвращаем старый k_systems
        k_systems = k_systems_temp
        return short_xs, short_ys, short_ts, size

    else:
        return 0


# Отображение начальных условий на единичную окружность
def coords_to_unit_circle(x, y):
    phi = np.arctan(y/x)

    if x < 0:
        phi = phi + np.pi

    # Unit Circle
    x_UC = np.cos(phi)
    y_UC = np.sin(phi)

    return x_UC, y_UC


# Сгенерировать НУ на единичной окружности
def IC_FHN_random_generator(path, doNeedShow=False, pathSave='0'):
    xs = []
    ys = []
    # Если файла нет, создаем
    if not exists(path):
        make_FHN_tr(path)

    IC = []
    xs, ys, size = read_FHN_coords_tr(path)
    for i in range(k_systems):
        # Рандомим координаты с ФХН
        randIndex = randint(0, len(xs)-1)
        x = xs[randIndex]
        y = ys[randIndex]

        # Записываем координаты на предельном цикле ФХН в НУ
        IC.append(x)
        IC.append(y)
        IC.append(z1_IC)
        IC.append(z2_IC)

    #plot_IC_unit_circle(IC, pathIC)

    # Рисуем НУ на траектории ФХН
    if doNeedShow or pathSave != '0':
        plt.plot(xs, ys)
        for i in range(k_systems):
            plt.scatter(IC[i * k], IC[i * k + 1], 150, label=str(i + 1))
        plt.legend()
        plt.grid()

        if pathSave:
            plt.savefig(pathSave)

        if doNeedShow:
            plt.show()
        plt.close()

    if doNeedShow:
        showInitialConditions(IC)

    return np.array(IC)

def read_FHN_coords_tr(path = FHN_coords_data_path):
    # Читаем из файла все координаты точек FHN
    with open(path, 'r') as f:
        f_data = f.readlines()

    size = int(f_data[0])

    xs = []
    ys = []
    # Выделяем x, y из строк
    for i in range(1, len(f_data)):
        line = f_data[i].split()
        xs.append(float(line[0]))
        ys.append(float(line[1]))
    return xs, ys, size

def plot_IC_FHN(IC, pathIC=0, pathFHN=FHN_coords_data_path):
    xs, ys, size = read_FHN_coords_tr(FHN_coords_data_path)

    plt.plot(xs, ys)
    for i in range(k_systems):
        plt.scatter(IC[i*k], IC[i*k+1], 150, label=str(i+1))
    plt.legend()

    if pathIC != 0:
        plt.savefig(pathIC)

    plt.grid()
    plt.show()
    return 0

# plot НУ на единичной окружности
def plot_IC_unit_circle(IC, pathIC=0):
    #fig, ax = plt.subplots(figsize=(5, 5))
    plt.Circle((0, 0), 1, fill=False)
    for i in range(k_systems):
        x = IC[i*k]
        y = IC[i*k + 1]
        x, y = coords_to_unit_circle(x, y)
        plt.scatter(x, y, 150, label=str(i+1))

    plt.legend()

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    if pathIC != 0:
        plt.savefig(pathIC)

    #plt.show()
    plt.close()

    return 0

# plot итогового состояния на единичной окружности
def plot_last_coords_unit_circle(delays, period, pathCoords=0):

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_circle = plt.Circle((0, 0), 1, fill=False)

    # Рисуем 1-й элемент
    plt.scatter(1.0, 0, 150, label=str(1))

    # Обходим каждый элемент
    for i in range(k_systems - 1):
        # Угол i-го элемента
        phi_i = 2 * np.pi * delays[i] / period

        x_i = np.cos(phi_i)
        y_i = np.sin(phi_i)

        plt.scatter(x_i, y_i, 150, label=str(i+2), marker=scatter_markers[i])

    ax.set_title('Итоговое состояние при G_inh=' + str(G_inh))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.legend()
    ax.add_artist(draw_circle)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    if pathCoords != 0:
        fig.savefig(pathCoords)
    #plt.show()
    plt.close()

    return 0


def generate_your_IC_FHN(arr_indexes_IC, pathIC=0, doNeedShow=False):
    if len(arr_indexes_IC) != k_systems:
        return 0
    xs, ys, size = read_FHN_coords_tr()

    for i in range(k_systems):
        if arr_indexes_IC[i] >= size or arr_indexes_IC[i] <= -size:
            return 0

    IC = []
    plt.plot(xs, ys)
    for i in range(k_systems):
        x = xs[arr_indexes_IC[i]]
        y = ys[arr_indexes_IC[i]]
        plt.scatter(x, y, 200, label=str(i+1), marker=scatter_markers[i])

        IC.append(x)
        IC.append(y)
        IC.append(z1_IC)
        IC.append(z2_IC)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Выбранные начальные условия')
    plt.grid()
    if pathIC:
        plt.savefig(pathIC)
    if doNeedShow:
        plt.show()
    plt.close()

    return np.array(IC)

####################################### Makers functions ######################################


# Проход сразу по многим g_inh с построением графиков
def make_investigation_of_dependence_on_inhibitory_coupling():
    stop = 10
    ex = joblib.Parallel(n_jobs=Nstreams)(
        joblib.delayed(graphics_4_with_G_inh)(step, initialConditions2) for step in range(0, stop))

    for k in range(0, stop):
        fig, axes = plt.subplots(2, 2)
        k_systems_2 = int(k_systems / 2)
        for i in range(0, k_systems_2):
            axes[0][i].plot(ex[k][1], ex[k][0][i], label=('eq' + str(i + 1)))
            axes[0][i].legend()
            axes[1][i].plot(ex[k][1], ex[k][0][i + k_systems_2], label=('eq' + str(i + k_systems_2 + 1)))
            axes[1][i].legend()

        fig.suptitle('G_inh = ' + str(ex[k][2]))
        plt.show()

def make_go_and_show_x_graphics(G_inh_, IC, tMax_, highAccuracy_=False, path_graph_x_start=0, path_graph_x_end=0,
                                path_graph_last_state=0, doNeedShow=False):
    global tMax, highAccuracy, G_inh, k_systems
    G_inh = G_inh_

    tMax = tMax_
    highAccuracy = highAccuracy_
    # Случайные начальные условия, которые будут одинаковы для всех экспериментов
    xs, ys, ts, G_inh = solveAndPlotWithIC(G_inh, 0, IC, path_graph_x_start, path_graph_x_end, doNeedShow)

    # Выбираем eq1 в качестве первого элемента, найдем период его колебаний
    # Трехмерный массив массив - 1) Номер нейрона; 2) Информация:
    # 1 - координата максимума, 2 - время максимума, 3 - индекс максимума
    inform_about_maximums = []
    for i in range(0, k_systems):
        inform_about_maximums.append(findMaximums(xs[i], ts))
        # print('maximums ' + str(i) + ': ' + str(inform_about_maximums[i][1]))

    # Теперь нужно рассмотреть, не подавлен ли какой элемент
    depressed_elements = []  # список подавленных элементов
    nondepressed_elem = 0
    for i in range(k_systems):
        if len(inform_about_maximums[i][0]) < 10:
            depressed_elements.append(i)
        else:
            nondepressed_elem = i

    return nondepressed_elem

# исследование зависимости параметров порядка от начальных условий
# исследование зависимости параметров порядка от начальных условий
def make_investigation_of_the_dependence_of_the_order_parameters_on_the_initial_conditions\
                (G_inh_, IC, tMax_, highAccuracy_=False,  path_graph_x_start=0, path_graph_x_end=0,
                path_graph_R=0, path_graph_last_state=0, doNeedShow=False):
    global tMax, highAccuracy, G_inh, k_systems
    G_inh = G_inh_

    tMax = tMax_
    highAccuracy = highAccuracy_
    # Случайные начальные условия, которые будут одинаковы для всех экспериментов
    xs, ys, ts, G_inh = solveAndPlotWithIC(G_inh, 0, IC, path_graph_x_start, path_graph_x_end, doNeedShow)

    # Выбираем eq1 в качестве первого элемента, найдем период его колебаний
    # Трехмерный массив массив - 1) Номер нейрона; 2) Информация:
    # 1 - координата максимума, 2 - время максимума, 3 - индекс максимума
    inform_about_maximums = []
    for i in range(0, k_systems):
        inform_about_maximums.append(findMaximums(xs[i], ts))
        # print('maximums ' + str(i) + ': ' + str(inform_about_maximums[i][1]))

    # Теперь нужно рассмотреть, не подавлен ли какой элемент
    depressed_elements = []     # список подавленных элементов
    nondepressed_elem = 0
    for i in range(k_systems):
        if len(inform_about_maximums[i][0]) < 10:
            depressed_elements.append(i)
        else:
            nondepressed_elem = i

    # Если подавлены все кроме одного, возвращаем R1,2 = 1 и заканчиваем
    if len(depressed_elements) == k_systems - 1:
        plt.figure()
        len_R = len(inform_about_maximums[nondepressed_elem][2])-2
        R1_arr = np.ones(len_R)
        R2_arr = np.ones(len_R)
        plt.plot(range(len_R), R1_arr, label='R1')
        plt.plot(range(len_R), R2_arr, label='R2')
        plt.title('Зависимость R1, R2 при G_inh = ' + str(G_inh))
        plt.xlabel('k')
        plt.ylabel('R1, R2')
        plt.legend()
        plt.grid()

        # Если передан путь, сохранить изображение
        if (path_graph_R != 0):
            plt.savefig(path_graph_R)

        return R1_arr, R2_arr, IC, depressed_elements

    # Удаляем подавленные элементы
    # 1) Создаем массив xs без подавленных элементов
    xs_no_depressed = []
    for i in range(len(xs)):
        xs_no_depressed.append(xs[i])
    for i in reversed(depressed_elements):
        xs_no_depressed.pop(i)
    # 2) Меняем k_systems, потому что он используется во множестве функций и
    #  с ним должны быть связаны размеры xs и прочие
    k_systems_with_depressed = k_systems
    k_systems -= len(depressed_elements)
    # 2) пересчитываем inform_about_maximums
    inform_about_maximums = []
    for i in range(0, len(xs_no_depressed)):
        inform_about_maximums.append(findMaximums(xs_no_depressed[i], ts))


    delay = []
    R1_arr = []
    R2_arr = []
    J_arr = []
    period = 0
    for j in range(10, len(inform_about_maximums[0][2]) - 2):
        delay_in_for = []
        delay_t = []
        for i in range(1, k_systems):
            # Находим период на текущем шаге
            period, i_period = findPeriod_i(inform_about_maximums[0], j)

            # Находим задержки на текущем шаге
            d, d_t = lagBetweenNeurons_3(inform_about_maximums[0][1], inform_about_maximums[0][2],
                                         inform_about_maximums[i][1], inform_about_maximums[i][2], period, j)
            delay_in_for.append(d)
            delay_t.append(d_t)

        for i in range(0, k_systems - 1):
            if abs(delay_in_for[i]) > period:
                if delay_in_for[i] > period:
                    delay_in_for[i] -= period
                elif delay_in_for[i] < period:
                    delay_in_for[i] += period
                i = 0
        delay.append(delay_in_for)

        # Находим параметр порядка
        R1, R2 = findOrderParam(period, delay_in_for)
        R1_arr.append(R1)
        R2_arr.append(R2)
        J_arr.append(j)

    # Классификация не рабочая
    #osc_type = classification(period, delay[-1])
    # if doNeedShow:
    #     print('delays:', delay[-1], 'T:', period, 'T/4: ', period / 4.0, 'T/2: ', period / 2.0, '3T/4: ',
    #         3.0 * period / 4.0)
        #print('type: ' + osc_type)

    # Графики параметров порядка
    plt.figure()
    plt.plot(J_arr, R1_arr, label='R1')
    plt.plot(J_arr, R2_arr, label='R2')
    plt.title('Зависимость R1, R2 при G_inh = ' + str(G_inh))
    plt.xlabel('k')
    plt.ylabel('R1, R2')
    plt.legend()
    plt.grid()

    # Если передан путь, сохранить изображение
    if(path_graph_R != 0):
        plt.savefig(path_graph_R)
    # Надо ли показывать график
    if doNeedShow:
        plt.show()
        plt.close()

    # Необходимо сохранить конечное состояние системы для вывода конечного графика
    last_state = []
    for i in range(k_systems):
        last_state.append(xs[i][-1])
        last_state.append(ys[1][-1])
        last_state.append(z1_IC)
        last_state.append(z2_IC)

    #showInitialConditions(last_state, 'Last state')
    if path_graph_last_state != 0:
        plot_last_coords_unit_circle(delay[-1], period, path_graph_last_state)
    plt.close()
    # Попытка показать весь график
    # margins = {  # +++
    #     "left": 0.020,
    #     "bottom": 0.060,
    #     "right": 0.990,
    #     "top": 0.990
    # }
    # plt.figure(figsize=(25, 5))
    # plt.subplots_adjust(**margins)
    # for i in range(len(xs)):
    #     plt.plot(ts, xs[i], label=('eq' + str(i + 1)), linestyle=plot_styles[i], color=plot_colors[i])
    #     plt.legend()
    # plt.grid()
    #
    # plt.show()
    k_systems = k_systems_with_depressed
    return R1_arr, R2_arr, IC, depressed_elements
    #return R1_arr, R2_arr, IC, osc_type
