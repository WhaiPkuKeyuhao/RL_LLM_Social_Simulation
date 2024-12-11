import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import normal_ad
from scipy import stats
from sklearn import metrics
import pmdarima as pm
import os
from week_data_process import week_tocsv, day_csv_to_week_csv

fileDir = '../data/train_0915aa_episode_3'
origin_csv = os.path.join(fileDir, "result_week.csv")

df = pd.read_csv(origin_csv, encoding='gbk')
d_true = df.iloc[0] ## 真实热力值
d_pred = df.iloc[1] ## 预测热力值

d_true_process = []
for idx, x in enumerate(d_true):
    d_true_process.append(x)

d_pred_process = []
for idx, x in enumerate(d_pred):
    if idx == 0 or x != 0:
        d_pred_process.append(x)
    else:
        d_pred_process.append(d_pred_process[idx - 1])

## 真实的数据和预测数据处理好



## 拿出3/4的数据做时间序列分析
train, valid = d_true_process[:int(len(d_true_process)*0.8)], d_true_process[int(len(d_true_process)*0.8):]

# define model
SARIMA_model = pm.auto_arima(train, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3,
                         m=24*12, #annual frequency(12 for month, 7 for week etc)
                         start_P=0,
                         seasonal=True, #set to seasonal
                         d=None,
                         D=1, #order of the seasonal differencing
                         trace=False,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)
print(SARIMA_model)

sarima_preds = []

for i in range(len(valid)):
    m_sarima = ARIMA(df_sp[:len(train) + i]['temp'], order=(1, 0, 0), seasonal_order=(0, 1, 2, 12)).fit()
    sarima_preds.append(m_sarima.forecast(1).values[0])

residuals = sorted([x - y for x, y in zip(sarima_preds, valid['temp'].values)])