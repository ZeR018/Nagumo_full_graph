import numpy as np
import math

import mainFunks as m
import time
import docx
import joblib
import matplotlib.pyplot as plt

Nstreams = 10
tMax = 500
highAccuracy = False
k_systems = 6

# Пути для сохранения данных
path_Doc = './Data/results/' + '7elems_UC_1' + '.docx'

R_data_path = './Data/r_data_protivofaza2.txt'
Graphic_data_path = './Data/graphics/saved_fig'
pathIC_full = './Data/graphics/savedIC_0.02.png'
pathIC = './Data/graphics/savedIC'
path_LS = './Data/graphics/finalState'
FHN_tr_path = './Data/FHW_coords.txt'

IC_A = np.array([
    1.538879577105834, -2.502158188845857, 0.01, 0.0,
    -2.502158188845857, 2.7191901965415948, 0.01, 0.0,
    2.7191901965415948, 1.724635452347087, 0.01, 0.0,
    1.724635452347087, 1.543752019655198, 0.01, 0.0,
    1.543752019655198, -2.4268647972584514, 0.01, 0.0,
    -2.4268647972584514, -0.9713907674897078, 0.01, 0.0
])

#################################################### Functions #################################################

def find_22_mode (G_inh, tMax, highAccuracy = False, doNeedShow = False):
    counter = 0
    while True:
        random_IC = m.IC_random_generator(-3, 3)
        R1_arr, R2_arr, IC = m. \
            make_investigation_of_the_dependence_of_the_order_parameters_on_the_initial_conditions \
            (G_inh, random_IC, tMax, highAccuracy, doNeedShow=doNeedShow)

        eps = 0.1
        if R1_arr[-1] < eps and R2_arr[-1] > 1 - eps:
            return counter, IC, R1_arr, R2_arr
        counter += 1

def find_22_mode_new (G_inh, tMax, highAccuracy = False, doNeedShow=False):
    counter = 0

    while True:
        random_IC = m.IC_FHN_random_generator(FHN_tr_path)
        R1_arr, R2_arr, IC = m.make_investigation_of_the_dependence_of_the_order_parameters_on_the_initial_conditions \
            (G_inh, random_IC, tMax, highAccuracy, doNeedShow=doNeedShow)

        eps = 0.1
        if R1_arr[-1] < eps and R2_arr[-1] > 1 - eps:
            return counter, IC, R1_arr, R2_arr
        counter += 1

# we had 6 elems system => will have 7_elems system with IC (a, b)
def add_elem_to_IC (IC, a, b):
    new_IC = []
    # Копируем старый массив
    for i in range(len(IC)):
        new_IC.append(IC[i])

    new_IC.append(a)
    new_IC.append(b)
    new_IC.append(m.z1_IC)
    new_IC.append(m.z2_IC)

    m.k_systems += 1
    return np.array(new_IC)


def remove_elem_from_IC(IC, num_elem=0):
    if num_elem > m.k_systems:
        print('ur idiot, i cant remove elem. num_elem > k_systems')
        return 0
    else:
        new_IC = []

        for i in range(m.k_systems):
            if i != num_elem:
                for k in range(0, 4):
                    new_IC.append(IC[i * m.k + k])

        m.k_systems -= 1
        return np.array(new_IC)


def make_investigation_with_changed_IC(index, doNeedShow=False):
    global Nstreams, tMax, highAccuracy
    print('Exp: ' + str(index))
    # Костыль
    m.k_systems = 7

    path_x = Graphic_data_path + '_x' + str(index) + '.png'
    path_R = Graphic_data_path + '_R' + str(index) + '.png'
    path_IC = pathIC + str(index) + '.png'
    path_last_state = path_LS + str(index) + '.png'

    G_inh = round(0.05 + 0.0002 * index, 6)
    # При маленьких значениях параметра связи берем большое время интегрирования
    if G_inh > - 0.005:
        tMax = 10000
    elif G_inh > - 0.02:
        tMax = 5000
    else:
        tMax = 200

    IC = m.IC_FHN_random_generator(FHN_tr_path, pathSave=path_IC)
    R1_arr, R2_arr, IC = m.make_investigation_of_the_dependence_of_the_order_parameters_on_the_initial_conditions\
        (G_inh, IC, tMax, highAccuracy, path_x, path_R, path_last_state, doNeedShow)

    return R1_arr, R2_arr, IC, path_x, path_R, path_IC, path_last_state, G_inh


############################################### Program ####################################################

def make_5_7_walk_to_Ginh():

    # Для проверки работы изменения НУ
    #IC_22 = IC_A
    # m.showInitialConditions(IC_22)

    # Убрать элемент
    #IC_22_size_5 = remove_elem_from_IC(IC_22, 2)
    #m.showInitialConditions(IC_22_size_5)

    # Добавить элемент
    # IC_22_size_7 = add_elem_to_IC(IC_22, 1.0, 1.0)
    # print(m.k_systems)
    # m.showInitialConditions(IC_22_size_7)

    # Находим какие-то новые НУ с режимом 2-2
    counter, IC_22, R1_arr_22, R2_arr_22 = find_22_mode(0.02, 500, doNeedShow= False)
    print('mode 2-2 found on the ' + str(counter) + ' attempt')
    m.showInitialConditions(IC_22)

    # G_inh changing params
    class G_inh_cp:
        start = 0.0
        stop = 0.09
        step = 0.01
        num = int((stop - start) / step)

    # Инициализируем файл doc
    mydoc = docx.Document()

    R1_arr = []
    R2_arr = []

    # Changing IC
    #IC_22_size_5 = remove_elem_from_IC(IC_22)
    IC_22_size_7 = add_elem_to_IC(IC_22, 1, -1)

    G_inh_arr = np.linspace(G_inh_cp.start, G_inh_cp.stop, G_inh_cp.num)
    for i in range(0, 5):
        loop_index = i * Nstreams + 1
        existance = joblib.Parallel(n_jobs=Nstreams)(joblib.delayed(make_investigation_with_changed_IC)
                        (IC_22_size_7, index / 1000.0)
                        for index in range(loop_index, loop_index + Nstreams))

        for j in range(len(existance)):
            R1_arr_i = existance[j][0]
            R2_arr_i = existance[j][1]
            IC = IC_22_size_7
            path_x = existance[j][2]
            path_R = existance[j][3]
            G_inh = existance[j][4]

            # Запись в файл .docx
            mydoc.add_heading('G_inh = ' + str(G_inh))
            mydoc.add_picture(path_x, width=docx.shared.Inches(8))
            mydoc.add_picture(path_R, width=docx.shared.Inches(5))
            mydoc.save(path_Doc)


def make_5_7_find_cyclop():
    maxCount = 5
    m.k_systems = 7
    # IC = []
    # # Генерируем НУ
    # for i in range(0, maxCount * Nstreams):
    #     m.k_systems = 7
    #     IC.append(m.IC_FHN_random_generator(FHN_tr_path))
    #     #m.showInitialConditions(IC[i])

    # Инициализируем файл doc
    mydoc = docx.Document()
    for i in range(0, maxCount):
        existance = joblib.Parallel(n_jobs=Nstreams)(joblib.delayed(make_investigation_with_changed_IC)
                        (index)
                        for index in range(i * Nstreams, i * Nstreams + Nstreams))

        for j in range(len(existance)):
            # R1_arr, R2_arr, path_x, path_R, path_IC, path_last_state, G_inh
            R1_arr_i = existance[j][0]
            R2_arr_i = existance[j][1]
            IC_i = existance[j][2]
            path_x = existance[j][3]
            path_R = existance[j][4]
            path_IC = existance[j][5]
            path_last_state = existance[j][6]
            G_inh = existance[j][7]


            # Запись в файл .docx
            mydoc.add_heading('Exp = ' + str(i * Nstreams + j) + ', G_inh = ' + str(G_inh), 2)
            print('k_systems:', m.k_systems, 'lenIC:', len(IC_i))
            for j in range(0, m.k_systems):
                mydoc.add_paragraph(str(IC_i[j * m.k]) + ', ' + str(IC_i[j * m.k + 1]) + ', ' +
                                    str(IC_i[j * m.k + 2]) + ', ' + str(IC_i[j * m.k + 3]) + ',')

            mydoc.add_picture(path_IC)
            mydoc.add_picture(path_last_state)
            mydoc.add_picture(path_x, width=docx.shared.Inches(8))
            mydoc.add_picture(path_R, width=docx.shared.Inches(5))
            mydoc.save(path_Doc)


#
# IC_33 = np.array([
#     1.7909191033523797, 0.3947975055180025, 0.01, 0,
#     -1.3196057166044584, -0.15643994894012603, 0.01, 0,
#     1.8357397674399445, 0.19450457790747858, 0.01, 0,
#     -1.9506185972198538, 0.9232371262393431, 0.01, 0,
#     1.5533010980534387, 0.014229690959855032, 0.01, 0,
#     -1.5336019316917493, 0.06607850268758791, 0.01, 0
# ])
# IC = m.IC_FHN_random_generator('./Data/FHW_coords.txt', pathIC)
# #m.plot_IC_FHN(IC)
# R1_arr, R2_arr, IC = m.make_investigation_of_the_dependence_of_the_order_parameters_on_the_initial_conditions(0.04, IC, 500, doNeedShow=False, path_graph_last_state=pathIC_full)
# #m.showInitialConditions(last_state, 'Last state')

make_5_7_find_cyclop()
