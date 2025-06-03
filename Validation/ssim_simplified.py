import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd

df = pd.read_csv("./data/true_sim.csv")
real_data = df["True"].to_numpy()
simulated_data = df["Sim"].to_numpy()


# 标准化数据到 [0, 255] 范围
def normalize_to_image(data):
    return data * 255

real_image = normalize_to_image(real_data)
simulated_image = normalize_to_image(simulated_data)

real_image = np.array(real_image).reshape(24, 7)
simulated_image = np.array(simulated_image).reshape(24, 7)

# 使用默认窗口大小 win_size=7（此时图像尺寸为 24x28，支持 7x7 窗口）
ssim_value = ssim(real_image, simulated_image, data_range=255,win_size=7, channel_axis=None)
print(f"SSIM 值: {ssim_value}")