# -*- coding: utf-8 -*-
"""
    @Project : py_project
    @File    : work9.py
    @Author  : Hongli Zhao
    @E-mail  : zhaohongli8711@outlook.com
    @Date    : 2024/7/6 下午9:31
    @Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from deap import base, creator, tools, algorithms
import random
import multiprocessing
from numba import jit

# 定义系统的传递函数
def system_dynamics(y, t, controller):
    theta, omega, alpha = y
    reference = 0.2  # 参考输入
    error = reference - theta
    controller.input['error'] = error
    controller.input['delta_error'] = -omega
    controller.compute()
    u = controller.output['output']
    dydt = [omega, alpha, -0.739 * omega - 0.921 * alpha + 1.151 * u + 0.1774]
    return dydt

fff = [0,0]
fff[0] = 0.5
fff[1] = 0.05

param_ranges = [
    (-fff[0], 0),
    (0, fff[0]),
    (0, fff[0]),
    (-fff[0], 0),
    (0, fff[0]),
    (0, fff[0]),
    (-fff[0], 0),
    (0, fff[0]),
    (0, fff[0]),
]

# 定义模糊控制器的生成函数
def create_fuzzy_controller(params):
    error = ctrl.Antecedent(np.arange(-fff[0], fff[0], fff[1]), 'error')
    delta_error = ctrl.Antecedent(np.arange(-fff[0], fff[0], fff[1]), 'delta_error')
    output = ctrl.Consequent(np.arange(-fff[0], fff[0], fff[1]), 'output')

    params = np.array(params)

    # 使用遗传算法优化的参数设置隶属函数
    error['NB'] = fuzz.trimf(error.universe, [-fff[0], -fff[0], params[0]])
    error['NS'] = fuzz.trimf(error.universe, [params[0], 0, 0])
    error['Z'] = fuzz.trimf(error.universe, [-params[1], 0, params[1]])
    error['PS'] = fuzz.trimf(error.universe, [0, 0, params[2]])
    error['PB'] = fuzz.trimf(error.universe, [params[2], fff[0], fff[0]])

    delta_error['NB'] = fuzz.trimf(delta_error.universe, [-fff[0], -fff[0], params[3]])
    delta_error['NS'] = fuzz.trimf(delta_error.universe, [params[3], 0, 0])
    delta_error['Z'] = fuzz.trimf(delta_error.universe, [-params[4], 0, params[4]])
    delta_error['PS'] = fuzz.trimf(delta_error.universe, [0, 0, params[5]])
    delta_error['PB'] = fuzz.trimf(delta_error.universe, [params[5], fff[0], fff[0]])

    output['NB'] = fuzz.trimf(output.universe, [-fff[0], -fff[0], params[6]])
    output['NS'] = fuzz.trimf(output.universe, [params[6], 0, 0])
    output['Z'] = fuzz.trimf(output.universe, [-params[7], 0, params[7]])
    output['PS'] = fuzz.trimf(output.universe, [0, 0, params[8]])
    output['PB'] = fuzz.trimf(output.universe, [params[8], fff[0], fff[0]])

    # 定义模糊规则
    rule1 = ctrl.Rule(error['NB'] & delta_error['NB'], output['NB'])
    rule2 = ctrl.Rule(error['NB'] & delta_error['NS'], output['NB'])
    rule3 = ctrl.Rule(error['NB'] & delta_error['Z'], output['NB'])
    rule4 = ctrl.Rule(error['NB'] & delta_error['PS'], output['NS'])
    rule5 = ctrl.Rule(error['NB'] & delta_error['PB'], output['Z'])

    rule6 = ctrl.Rule(error['NS'] & delta_error['NB'], output['NB'])
    rule7 = ctrl.Rule(error['NS'] & delta_error['NS'], output['NB'])
    rule8 = ctrl.Rule(error['NS'] & delta_error['Z'], output['NS'])
    rule9 = ctrl.Rule(error['NS'] & delta_error['PS'], output['Z'])
    rule10 = ctrl.Rule(error['NS'] & delta_error['PB'], output['PS'])

    rule11 = ctrl.Rule(error['Z'] & delta_error['NB'], output['NS'])
    rule12 = ctrl.Rule(error['Z'] & delta_error['NS'], output['NS'])
    rule13 = ctrl.Rule(error['Z'] & delta_error['Z'], output['Z'])
    rule14 = ctrl.Rule(error['Z'] & delta_error['PS'], output['PS'])
    rule15 = ctrl.Rule(error['Z'] & delta_error['PB'], output['PS'])

    rule16 = ctrl.Rule(error['PS'] & delta_error['NB'], output['NS'])
    rule17 = ctrl.Rule(error['PS'] & delta_error['NS'], output['Z'])
    rule18 = ctrl.Rule(error['PS'] & delta_error['Z'], output['PS'])
    rule19 = ctrl.Rule(error['PS'] & delta_error['PS'], output['PB'])
    rule20 = ctrl.Rule(error['PS'] & delta_error['PB'], output['PB'])

    rule21 = ctrl.Rule(error['PB'] & delta_error['NB'], output['Z'])
    rule22 = ctrl.Rule(error['PB'] & delta_error['NS'], output['PS'])
    rule23 = ctrl.Rule(error['PB'] & delta_error['Z'], output['PB'])
    rule24 = ctrl.Rule(error['PB'] & delta_error['PS'], output['PB'])
    rule25 = ctrl.Rule(error['PB'] & delta_error['PB'], output['PB'])

    # 增加默认规则，确保至少有一个规则被激活
    rule_default = ctrl.Rule(error['Z'] & delta_error['Z'], output['Z'])

    # 创建控制系统
    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                         rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
                                         rule20, rule21, rule22, rule23, rule24, rule25, rule_default])

    # 创建控制器
    controller = ctrl.ControlSystemSimulation(control_system)
    return controller

# 使用Numba加速适应度函数
@jit(nopython=True)
def calculate_fitness(theta, t):
    target = 0.2
    max_theta = np.max(theta)
    overshoot = (max_theta - target) / target * 100

    ninety_percent_index = np.where(theta >= 0.9 * target)[0]
    rise_time = t[ninety_percent_index[0]] - t[0] if len(ninety_percent_index) > 0 else 999999

    steady_state_indices = np.where(np.abs(theta - target) <= target * 0.02)[0]
    steady_state_time = 999999
    if len(steady_state_indices) > 0:
        for idx in steady_state_indices:
            if np.all(np.abs(theta[idx:] - target) <= target * 0.02):
                steady_state_time = t[idx]
                if steady_state_time >= 25:
                    steady_state_time = 999999
                break

    # steady_state_error = np.abs(theta[-1] - target) / target * 100
    # if steady_state_time >= 25:
    #     steady_state_error = 999999

    # fitness_value = 1 * overshoot + 6 * rise_time + 1 * steady_state_time + 4 * steady_state_error
    fitness_value = 1 * overshoot + 50 * rise_time + 1 * steady_state_time
    return fitness_value

# 定义适应度函数
def fitness(params):
    try:
        controller = create_fuzzy_controller(params)
    except AssertionError:
        return float('inf'),  # 返回一个非常大的值，表示适应度很差

    y0 = [0, 0, 0]
    t = np.linspace(0, 30, 300)
    response = odeint(system_dynamics, y0, t, args=(controller,))
    theta = response[:, 0]

    fitness_value = calculate_fitness(theta, t)
    return fitness_value,

# 遗传算法设置
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 自定义初始化函数
def init_individual(icls, param_ranges):
    individual = icls(random.uniform(a, b) for a, b in param_ranges)
    return individual

# 注册自定义初始化函数
toolbox.register("individual", init_individual, creator.Individual, param_ranges)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 遗传算法操作
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("evaluate", fitness)

if __name__ == '__main__':
    # 并行化
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # 遗传算法参数
    population = toolbox.population(n=30)
    initial_values = [-0.40600203006476726, 0.2165269405773259, 0.33239149440300386, -0.06109542626659414, 0.020992008085260153, 0.19366324693194947, -0.4229821445161332, 0.08889105822727436, 0.22189308048913164]
    initial_values2 = [-0.4324519357420834, 0.24322807467086105, 0.33239149440300386, -0.061095426266594144, 0.030906263946451216, 0.19366324693194947, -0.4180467606678175, 0.08889105822727436, 0.2918407091769414]
    initial_values3 = [-0.4235709009045971, 0.24225452538559916, 0.33239149440300386, -0.06424658804459948, 0.03158854082815961, 0.1951971718640115, -0.4224272444733286, 0.08889105822727436, 0.35210913156246376]
    initial_values4 = [-0.4114293926077724, 0.21652694057732597, 0.3058151401336632, -0.061095426266594144, 0.020992008085260153, 0.1818927198438243, -0.42298214451613314, 0.08889105822727436, 0.22189308048913164]
    initial_values5 = [-0.32745911211215284, 0.24230945320313854, 0.33239149440300386, -0.0642991680888633, 0.031435426121412716, 0.19459045397623986, -0.4224914775362006, 0.08873093713024269, 0.35193231548358983]
    initial_values6 = [-0.32750175560008177, 0.24230945320313846, 0.33239149440300386, -0.06387574585411795, 0.031435426121412716, 0.20236233905446438, -0.425097639691571, 0.0716831715481494, 0.4135509524674473]
    initial_values7 = [-0.11937703864433051, 0.24302122577190866, 0.3843769125765124, -0.06265264408566826, 0.04336982002102204, 0.43441522371923197, -0.4198687807724506, 0.08885267426301538, 0.34331498378620423]
    initial_values8 =[-0.4594428905875751, 0.2430212257719087, 0.42287377964471395, -0.06265264408566826, 0.043370447313442986, 0.4444292632413226, -0.4198687807724507, 0.08885267426301535, 0.3429942453667024]
    initial_values9 =[-0.121853086302474, 0.24297985914333248, 0.3980835040988893, -0.06265300321665083, 0.04346136755527576, 0.4324658983262635, -0.41975131224775175, 0.08885287630011682, 0.3407847502122957]
    initial_values10 = [-0.4880838780437378, 0.2430212257719087, 0.4357864319759265, -0.06265264408566826, 0.04311298486796782, 0.4483828092564216, -0.4198687807724508, 0.08885267426301535, 0.33831614910600555]
    initial_values11 = [-0.41839061764627533, 0.21652694057732608, 0.2950022162770245, -0.06109542626659416, 0.021014491166976818, 0.494832441240838, -0.4231207405420739, 0.08889105822727436, 0.22185019178119392]
    population[0][:] = initial_values
    population[1][:] = initial_values2
    population[2][:] = initial_values3
    population[3][:] = initial_values4
    population[4][:] = initial_values5
    population[5][:] = initial_values6
    population[6][:] = initial_values7
    population[7][:] = initial_values8
    population[8][:] = initial_values9
    population[9][:] = initial_values10
    population[10][:] = initial_values11
    ngen = 50
    cxpb = 0.5
    mutpb = 0.4

    # 运行遗传算法
    result, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

    # 获取最优解
    best_individual = tools.selBest(result, k=1)[0]
    print("Best Individual:", best_individual)

    # 使用最优参数创建模糊控制器并仿真
    best_controller = create_fuzzy_controller(best_individual)
    y0 = [0, 0, 0]
    t = np.linspace(0, 30, 300)
    response = odeint(system_dynamics, y0, t, args=(best_controller,))
    theta = response[:, 0]

    # 计算超调量
    max_theta = np.max(theta)
    overshoot = (max_theta - 0.2) / 0.2 * 100

    # 计算上升时间（到50%）
    ninety_percent_index = np.where(theta >= 0.5 * 0.2)[0]
    rise_time = t[ninety_percent_index[0]] - t[0] if len(ninety_percent_index) > 0 else np.nan

    # 计算稳态时间（进入并保持在2%范围内）
    steady_state_indices = np.where(np.abs(theta - 0.2) <= 0.2 * 0.02)[0]
    steady_state_time = np.nan
    if len(steady_state_indices) > 0:
        for idx in steady_state_indices:
            if np.all(np.abs(theta[idx:] - 0.2) <= 0.2 * 0.02):
                steady_state_time = t[idx]
                break
    if steady_state_time >= 25:
        steady_state_time = np.nan

    # 计算稳态误差
    steady_state_error = np.abs(theta[-1] - 0.2) / 0.2 * 100
    if steady_state_time == np.nan:
        steady_state_error = np.nan

    # 打印结果
    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Rise Time: {rise_time:.2f} seconds")
    print(f"Steady State Time: {steady_state_time:.2f} seconds")
    print(f"Steady State Error: {steady_state_error:.2f}%")

    # 绘制最优模糊控制器的阶跃响应
    plt.figure()
    plt.plot(t, theta, label='Theta (rad)')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Reference (0.2 rad)')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.title('Step Response with Optimized Fuzzy Controller')
    plt.legend()
    plt.grid(True)
    plt.show()
