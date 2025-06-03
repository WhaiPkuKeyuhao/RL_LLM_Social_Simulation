import pandas as pd
from math import log
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# f = open(f"data/power_usage/power_usage_981_1.txt", 'r', encoding="utf-8")
# lines = f.readlines()
# sim_data = eval(lines[1])
#
# real_df = pd.read_csv("data/power_usage/target_cluster_2_500.csv")
# real_data = real_df.iloc[0].tolist()[:len(sim_data)]

df = pd.read_csv("./data/true_sim.csv")
real_data = df["True"].to_list()
sim_data = df["Sim"].to_list()


sim_sum = sum(sim_data)
real_sum = sum(real_data)

sim_data = [i/sim_sum for i in sim_data]
real_data = [i/real_sum for i in real_data]





kl_div = 0
for real , sim in zip(real_data,sim_data):
    kl_div += real*log(real/sim)

print(f"The KL-Divergence of phone usage is {kl_div}")


# 找到一组类似的分布对比达成一样的KL散度

# 定义域：1到24*7的整数，表示一周的数据点，每个数据点代表一个小时
x = np.arange(0, 24 * 7)
base_value = 0  # 基础值，确保没有负值

# 定义初始设定的高斯函数的参数
peak_times = [8, 22]  # 峰值出现的时间点（小时）
amplitudes = [10, 20]  # 对应峰值的振幅
sigmas = [2, 2]  # 对应峰值的高斯函数标准差（控制峰值的宽度）


# 定义偏移后的高斯函数的参数
peak_times_shifted = [7.2, 21.2]  # 峰值出现的时间点（小时）
amplitudes_shifted = [8.5, 18.5]  # 对应峰值的振幅
sigmas_shifted = [1.8, 1.85]  # 对应峰值的高斯函数标准差（控制峰值的宽度）


# 构造初始的y数组
y_base = np.full_like(x, base_value)
y_base = y_base.astype(float)
# 为每个峰值添加高斯函数
for peak_time, amplitude, sigma in zip(peak_times, amplitudes, sigmas):
    # 计算每个时间点与峰值时间点的差异（以小时为单位）
    time_difference = (x % 24) - peak_time
    # 计算高斯函数值，并确保它不会低于0（虽然对于正态分布来说这是不必要的）
    gaussian_peak = amplitude * norm.pdf(time_difference, loc=0, scale=sigma)
    # 将高斯峰值加到y数组上
    y_base += np.where(gaussian_peak < 0, 0, gaussian_peak)


# 构造偏移后的y数组
y_shifted = np.full_like(x, base_value)
y_shifted = y_shifted.astype(float)
# 为每个峰值添加高斯函数
for peak_time, amplitude, sigma in zip(peak_times_shifted, amplitudes_shifted, sigmas_shifted):
    # 计算每个时间点与峰值时间点的差异（以小时为单位）
    time_difference = (x % 24) - peak_time
    # 计算高斯函数值，并确保它不会低于0（虽然对于正态分布来说这是不必要的）
    gaussian_peak = amplitude * norm.pdf(time_difference, loc=0, scale=sigma)
    # 将高斯峰值加到y数组上
    y_shifted += np.where(gaussian_peak < 0, 0, gaussian_peak)

shifted_sum = sum(y_shifted)
base_sum = sum(y_base)

y_shifted = [i/shifted_sum for i in y_shifted]
y_base = [i/base_sum for i in y_base]

# 绘图
plt.figure(figsize=(14, 7))
plt.plot(x, y_base, label='Base Distribution',color = "blue")
plt.plot(x, y_shifted, label='Shifted Distribution',color = "red")
plt.xlabel('Hour of the Week')
plt.ylabel('Value')
plt.xticks(np.arange(1, 24 * 7 + 1, 24), [f'Day {i + 1}' for i in range(7)], rotation=45)
plt.xticks(np.arange(1, 24 * 7 + 1, 1), minor=True)
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("./images/example_of_similar_distribution")



kl_div = 0
for real , sim in zip(y_base,y_shifted):
    kl_div += real*log(real/sim)

print(f"The KL-Divergence of example distribution is {kl_div}")

