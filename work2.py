# -*- coding: utf-8 -*-
"""
    @Project : py_project
    @File    : work2.py
    @Author  : Hongli Zhao
    @E-mail  : zhaohongli8711@outlook.com
    @Date    : 2024/7/6 下午6:57
    @Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import control as ctl

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 系统的开环传递函数
num = [1.151, 0.1774]
den = [1, 0.739, 0.921, 0]
sys = ctl.TransferFunction(num, den)

# 性能指标计算函数
def performance(params):
    Kp, Ki, Kd = params
    if Kp >=20 or Ki>=20 or Kd >= 20:
        return 9999999999
    # 设计PID控制器
    C = ctl.TransferFunction([Kd, Kp, Ki], [1, 0])
    # 闭环系统传递函数
    T = ctl.feedback(C*sys, 1)
    # 仿真系统响应
    t = np.linspace(0, 20, 1000)
    _, y = ctl.step_response(0.2*T, t)
    # 性能指标计算
    target = 0.2
    theta = y
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
                if steady_state_time >= 15:
                    steady_state_time = 999999
                break

    # 目标函数
    return overshoot + rise_time + steady_state_time

# 优化PID参数
initial_guess = [10, 1, 1]
result = minimize(performance, initial_guess, method='Nelder-Mead')
Kp_opt, Ki_opt, Kd_opt = result.x

print(f'优化后的PID参数: Kp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f}')

# 使用优化后的参数重新计算阶跃响应
C_opt = ctl.TransferFunction([Kd_opt, Kp_opt, Ki_opt], [1, 0])
T_opt = ctl.feedback(C_opt*sys, 1)
t = np.linspace(0, 40, 1000)
t, y = ctl.step_response(0.2*T_opt, t)

# 绘制优化后的阶跃响应曲线
plt.plot(t, y)
plt.title('优化后的闭环系统阶跃响应')
plt.xlabel('时间 (秒)')
plt.ylabel('输出 (弧度)')
plt.grid()
plt.show()


theta = y
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
steady_state_indices = np.where(np.abs(theta - 0.2) <= 0.2*0.02)[0]

steady_state_time = np.nan  # 默认值

if len(steady_state_indices) > 0:
    # 遍历找到进入并保持在稳态范围内的时间点
    for idx in steady_state_indices:
        # 检查从该点开始到结束的所有点是否都在稳态范围内
        if np.all(np.abs(theta[idx:] - 0.2) <= 0.2*0.02):
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