# -*- coding: utf-8 -*-
"""
    @Project : py_project
    @File    : work4.py
    @Author  : Hongli Zhao
    @E-mail  : zhaohongli8711@outlook.com
    @Date    : 2024/7/6 下午7:13
    @Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 定义模糊控制器的输入和输出变量
error = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'error')
delta_error = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'delta_error')
output = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'output')

# 定义模糊集
error['NB'] = fuzz.trimf(error.universe, [-1, -1,-0.5])
error['NS'] = fuzz.trimf(error.universe, [-1, -0.5, 0])
error['Z'] = fuzz.trimf(error.universe, [-0.5, 0, 0.5])
error['PS'] = fuzz.trimf(error.universe, [0, 0.5, 1])
error['PB'] = fuzz.trimf(error.universe, [0.5, 1, 1])

delta_error['NB'] = fuzz.trimf(delta_error.universe, [-1, -1, -0.5])
delta_error['NS'] = fuzz.trimf(delta_error.universe, [-1, -0.5, 0])
delta_error['Z'] = fuzz.trimf(delta_error.universe, [-0.5, 0, 0.5])
delta_error['PS'] = fuzz.trimf(delta_error.universe, [0, 0.5, 1])
delta_error['PB'] = fuzz.trimf(delta_error.universe, [0.5, 1, 1])

output['NB'] = fuzz.trimf(output.universe, [-1, -1, -0.5])
output['NS'] = fuzz.trimf(output.universe, [-1, -0.5, 0])
output['Z'] = fuzz.trimf(output.universe, [-0.5, 0, 0.5])
output['PS'] = fuzz.trimf(output.universe, [0, 0.5, 1])
output['PB'] = fuzz.trimf(output.universe, [0.5, 1, 1])

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

# 创建控制系统
control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                     rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
                                     rule20, rule21, rule22, rule23, rule24, rule25])

# 创建控制器
controller = ctrl.ControlSystemSimulation(control_system)

# 定义系统的传递函数
def system_dynamics(y, t, controller):
    theta, omega, alpha = y
    reference = 0.2  # 参考输入
    error = reference - theta
    controller.input['error'] = error
    controller.input['delta_error'] = -omega
    controller.compute()
    u = controller.output['output']
    dydt = [omega, alpha, -0.739*omega - 0.921*alpha + 1.151*u + 0.1774]
    return dydt

# 初始条件
y0 = [0, 0, 0]
t = np.linspace(0, 40, 400)

# 使用odeint进行仿真
response = odeint(system_dynamics, y0, t, args=(controller,))

# 提取响应的角度
theta = response[:, 0]

# 计算超调量
max_theta = np.max(theta)
overshoot = (max_theta - 0.2) / 0.2 * 100

# 计算上升时间（从10%到90%）
# 找到首次达到90%目标值的索引
ninety_percent_index = np.where(theta >= 0.9*0.2 )[0]

if len(ninety_percent_index) > 0:
    # 计算从开始到首次达到90%目标值的时间
    rise_time = t[ninety_percent_index[0]] - t[0]
else:
    rise_time = np.nan

# 计算稳态时间（进入并保持在2%范围内）
# 找到进入稳态范围的索引
steady_state_indices = np.where(np.abs(theta - theta[-1]) <= theta[-1]*0.02)[0]

steady_state_time = np.nan  # 默认值

if len(steady_state_indices) > 0:
    # 遍历找到进入并保持在稳态范围内的时间点
    for idx in steady_state_indices:
        # 检查从该点开始到结束的所有点是否都在稳态范围内
        if np.all(np.abs(theta[idx:] - theta[-1]) <= theta[-1]*0.02):
            steady_state_time = t[idx]
            break  # 找到第一个满足条件的时间点后退出循环
else:
    steady_state_time = np.nan

# 计算稳态误差
steady_state_error = np.abs(theta[-1] - 0.2) / 0.2 * 100

# 打印结果
print(f"Overshoot: {overshoot:.2f}%")
print(f"Rise Time: {rise_time:.2f} seconds")
print(f"Steady State Time: {steady_state_time:.2f} seconds")
print(f"Steady State Error: {steady_state_error:.2f}%")

# 绘制阶跃响应
plt.figure()
plt.plot(t, theta, label='Theta (rad)')
plt.axhline(y=0.2, color='r', linestyle='--', label='Reference (0.2 rad)')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Step Response with Fuzzy Controller')
plt.legend()
plt.grid(True)
plt.show()

