# -*- coding: utf-8 -*-
"""
    @Project : py_project
    @File    : work.py
    @Author  : Hongli Zhao
    @E-mail  : zhaohongli8711@outlook.com
    @Date    : 2024/7/6 下午6:36
    @Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ctl
import matplotlib.font_manager as fm

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 系统的开环传递函数
num = [1.151, 0.1774]
den = [1, 0.739, 0.921, 0]
sys = ctl.TransferFunction(num, den)

# PID控制器参数
Kp = 10  # 设定初始值
Ki = 5   # 设定初始值
Kd = 5   # 设定初始值

# 设计PID控制器
C = ctl.TransferFunction([Kd, Kp, Ki], [1, 0])

# 闭环系统传递函数
T = ctl.feedback(C*sys, 1)

# 仿真系统响应
t = np.linspace(0, 20, 1000)
t, y = ctl.step_response(0.2*T, t)

# 绘制阶跃响应曲线
plt.plot(t, y)
plt.title('闭环系统的阶跃响应')
plt.xlabel('时间 (秒)')
plt.ylabel('输出 (弧度)')
plt.grid()
plt.show()

# 性能指标计算
overshoot = (max(y) - 0.2) / 0.2 * 100
rise_time = t[next(i for i, v in enumerate(y) if v >= 0.2)]
settling_time = t[next(i for i, v in enumerate(y) if abs(v - 0.2) <= 0.02)]

print(f'超调量: {overshoot:.2f}%')
print(f'上升时间: {rise_time:.2f}秒')
print(f'稳态时间: {settling_time:.2f}秒')
print(f'稳态误差: {abs(y[-1] - 0.2)/0.2 * 100:.2f}%')

theta = y
# 计算超调量
max_theta = np.max(theta)
overshoot = (max_theta - 0.2) / 0.2 * 100

# 计算上升时间
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
