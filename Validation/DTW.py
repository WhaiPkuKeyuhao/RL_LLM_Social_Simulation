import numpy as np
import pandas as pd
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv("./data/true_sim.csv")
real_data = df["True"].to_numpy()
simulated_data = df["Sim"].to_numpy()


d, paths = dtw.warping_paths(real_data, simulated_data, window=25, psi=2)
best_path = dtw.best_path(paths)
plt.figure(figsize=(15,13))
dtwvis.plot_warpingpaths(real_data, simulated_data, paths, best_path)
plt.savefig("./images/DTW_best_path")

best_path_df = pd.DataFrame(best_path, columns=['real_data_index', 'simulated_data_index'])
best_path_df.to_csv("./data/DTW_best_path_points.csv", index=False)

x = best_path_df["real_data_index"].to_list()
y_target = best_path_df["simulated_data_index"].to_list()
y_pred = [i for i in x]


plt.figure(figsize=(6,6))
# 添加label参数用于图例标识
plt.plot(x, y_pred, color="red", lw=3, label='Regression Line')
plt.scatter(x, y_target, color="blue", label='Label', s = 10)

# 添加标题和轴标签（根据实际含义替换占位符）
plt.title("Regression Result of DTW Optimal Path", fontsize=16, pad=20)  # 标题[9,10](@ref)
plt.xlabel("Time Index of Real Data", fontsize=12)  # 替换为实际X轴含义[6,7](@ref)
plt.ylabel("Time Index of Simulated Data", fontsize=12)  # 替换为实际Y轴含义[6,7](@ref)

# 添加图例（自动选择最佳位置）
plt.legend(loc='best', fontsize=12, frameon=True)  # 图例设置[1,4](@ref)

plt.text(
    0.95, 0.05,  # 右下角位置（坐标轴比例）
    "R_square = 0.99",
    transform=plt.gca().transAxes,  # 使用坐标轴坐标系
    ha='right',  # 水平右对齐
    va='bottom',  # 垂直底对齐
    fontsize=14,
    fontweight='bold',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')  # 白色半透背景框
)

# 添加网格线提高可读性（可选）
plt.grid(alpha=0.3)

plt.savefig("./images/DTW_regression.png")

r2_score  = r2_score(y_target, y_pred)
print(f"DWT基线R方为{r2_score}")
