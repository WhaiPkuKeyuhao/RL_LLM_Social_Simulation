import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data_df = pd.read_csv("data/true_sim.csv")
sim_data = data_df["Sim"].tolist()
real_data = data_df["True"].tolist()

# 去趋势化
real_data_detrend = signal.detrend(real_data, type='linear')
sim_data_detrend = signal.detrend(sim_data, type='linear')

# 归一化
scaler = StandardScaler()
real_data_scaled = scaler.fit_transform(real_data_detrend.reshape(-1, 1)).flatten()
sim_data_scaled = scaler.transform(sim_data_detrend.reshape(-1, 1)).flatten()

# 获取频域信息
# 采样频率 = 1小时 = 24次1天
fs = 24  # 样本/天
n = len(real_data_scaled)
freqs = np.fft.fftfreq(n, d=1/fs)  # 频率轴（单位：周期/天）
fft_real = np.fft.fft(real_data_scaled)
fft_sim = np.fft.fft(sim_data_scaled)

# 定位主频
# 计算功率谱密度
power_real = np.abs(fft_real) ** 2
power_sim = np.abs(fft_sim) ** 2
# 找到主频（排除直流分量）
main_freq_real = freqs[np.argmax(power_real[1:]) + 1]  # +1跳过0Hz
main_freq_sim = freqs[np.argmax(power_sim[1:]) + 1]
# 通常主频应为1周期/天（频率=1）

# 找到日周期对应的索引
target_freq = 1  # 1周期/天
idx = np.where(np.isclose(freqs, target_freq, atol=1e-3))[0][0]
# 获取相位（弧度）
phase_real = np.angle(fft_real[idx])
phase_sim = np.angle(fft_sim[idx])
# 计算相位差（弧度）
phase_diff = phase_sim - phase_real

# 一个周期对应的时间（天）
period = 1 / target_freq  # 1天
# 相位差转换为小时
time_diff_hours = (phase_diff / (2 * np.pi)) * period * 24
print(f"模拟数据相对真实数据的相位差：{time_diff_hours:.2f}小时")

target_freq_half = 2  # 2周期/天（12小时一次）
idx_half = np.where(np.isclose(freqs, target_freq_half, atol=1e-3))[0][0]
phase_real_half = np.angle(fft_real[idx_half])
phase_sim_half = np.angle(fft_sim[idx_half])
# 计算时间差
time_diff_half = ( (phase_sim_half - phase_real_half) / (2 * np.pi) ) * 12  # 12小时周期
print(f"半日周期相位差：{time_diff_half:.2f}小时")

plt.figure(figsize=(20, 15))
# 功率谱
plt.subplot(2, 1, 1)
plt.plot(freqs[:n//2], power_real[:n//2], label='Real')
plt.plot(freqs[:n//2], power_sim[:n//2], label='Sim')
plt.axvline(target_freq, color='r', linestyle='--', label='1 cycle/day')
plt.axvline(target_freq_half, color='g', linestyle='--', label='2 cycles/day')
plt.xlabel('Frequency (cycles/day)')
plt.ylabel('Power')
plt.legend()
# 相位差
plt.subplot(2, 1, 2)
plt.bar(['Daily', 'Half-Day'], [time_diff_hours, time_diff_half])
plt.ylabel('Time Difference (hours)')
plt.title('Phase Shift between Real and Simulated Data')
plt.show()