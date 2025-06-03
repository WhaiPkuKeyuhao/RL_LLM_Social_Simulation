import numpy as np
import pandas as pd
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt

# f = open(f"data/power_usage/power_usage_981_1.txt", 'r', encoding="utf-8")
# lines = f.readlines()
# simulated_data = np.array(eval(lines[1]))
#
# real_df = pd.read_csv("data/power_usage/target_cluster_2_500.csv")
# real_data = np.array(real_df.iloc[0].tolist()[:len(simulated_data)])

df = pd.read_csv("./data/true_sim.csv")
real_data = df["True"].to_numpy()
simulated_data = df["Sim"].to_numpy()


d, paths = dtw.warping_paths(real_data, simulated_data, window=25, psi=2)
best_path = dtw.best_path(paths)
plt.figure(figsize=(30,30))
dtwvis.plot_warpingpaths(real_data, simulated_data, paths, best_path)
plt.savefig("./images/DTW_best_path")