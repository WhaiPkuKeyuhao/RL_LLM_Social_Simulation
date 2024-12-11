import sys
sys.path.append(".")
sys.path.append("../")

import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

WEEK_HOLIDAY_TYPES = [
    "周一工作日",
    "周二工作日",
    "周三工作日",
    "周四工作日",
    "周五工作日",
    "周六节假日",
    "周日节假日"
]

WEEK_HOLIDAY_TYPES_EN = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

def day_csv_to_week_csv():
    origin_csv = os.path.join("data", "out.csv")
    target_csv = os.path.join("data", "out_week.csv")

    df=pd.read_csv(origin_csv,encoding='gbk')

    result = {}
    for idx, name in enumerate(WEEK_HOLIDAY_TYPES):
        d_true = df[df['type']==name]
        d_true_value = d_true[d_true.columns[1:]]
        for i,x in enumerate(d_true_value.iloc[0]):
            if x.split('%')[0]!='----':
                name = "{}|{}".format(WEEK_HOLIDAY_TYPES_EN[idx], d_true_value.columns[i])
                value = float(x.split('%')[0])/100
                result[name] = [value]

    result_df = pd.DataFrame(result)
    result_df.to_csv(target_csv, index=False,encoding='gbk')


def week_tocsv(fileDir, fileName, numAgent):
    origin_csv = os.path.join("data", "out_week.csv")
    target_csv = os.path.join(fileDir, "result_week.csv")

    count_play = {}
    for idx, name in enumerate(WEEK_HOLIDAY_TYPES_EN):
        filePath = os.path.join(fileDir, f"{fileName}_{idx + 1}.txt")

        f = open(filePath, "r")
        lines = f.readlines()

        person_ids=[]
        history_of_person=[]

        for i, line in enumerate(lines):
            if i % 2 == 0:
                person_ids.append(line.strip())
            else:
                history_of_person.append(json.loads(line.strip()))

        df=pd.DataFrame({'person_ids':pd.Series(person_ids)[-numAgent:], 'history_of_person':pd.Series(history_of_person)[-numAgent:]})
        df=df.reset_index(drop=True)

        count={}
        def function(r, minute):
            count.setdefault(f'{minute//60}:{minute%60:02d}', 0)
            for d in r['history_of_person']:
                if d['start_time'] <= minute and minute < d['end_time']:
                    #print(d['location'])
                    if (d['location']=='在家里' and d['action'] in ("放松", "阅读")) or \
                    (d['location'] in ('在小区商场', '在小区公共场所') and d['action'] in ("运动", "购物")):
                        count[f'{minute//60}:{minute%60:02d}']+=1

        for minute in range(0,24*60,5):
            for r in range(len(df)):#每行
                function(df.iloc[r], minute)

        for k in count:
            count_play["{}|{}".format(name, k)] = [count[k]/(numAgent)]
    count_play=pd.DataFrame(count_play)

    d=pd.read_csv(origin_csv,encoding='gbk')
    d=pd.concat((d, count_play))
    d.to_csv(target_csv, index=False,encoding='gbk')

def week_plot(fileDir):
    origin_csv = os.path.join(fileDir, "result_week.csv")

    df = pd.read_csv(origin_csv,encoding='gbk')
    d_true = df.iloc[0]
    d_pred = df.iloc[1]

    d_true_process = []
    for idx, x in enumerate(d_true):
        d_true_process.append(x)
    
    d_pred_process = []
    for idx, x in enumerate(d_pred):
        if idx == 0 or x != 0:
            d_pred_process.append(x)
        else:
            d_pred_process.append(d_pred_process[idx - 1])


    time_interval = 2
    d_x = []
    for idx, x in enumerate(list(df.columns)):
        day = x.split("|")[0]
        hour = x.split("|")[1].split(":")[0]

        hour = int(float(hour) // time_interval * time_interval)
        d_x.append(f"{day}-{hour}h")

    plt.figure(figsize=(24 * (4 / time_interval), 15))
    plt.xticks(rotation=70)
    plt.rcParams ["font.family"] = "SimHei"
    df = pd.DataFrame({'time':d_x, 'value':d_true_process, 'pred': d_pred_process})
    f=sns.lineplot(x='time', y='value', data=df, label='value')
    f=sns.lineplot(x='time', y='pred', data=df, label='pred')
    f.get_figure().savefig(os.path.join(fileDir, f'一周热力值拟合.png'), dpi = 100)
    #df.plot()


if __name__ == "__main__":
    #day_csv_to_week_csv()

    DIR_NAME = "data\\train_0930a"
    FILE_NAME = "data_episode_2"
    AGENT_NUMS = 512
    #RESIDENT_PATH = "config/agentSimulate10000.json"

    #week_tocsv(DIR_NAME, FILE_NAME, AGENT_NUMS,)
    week_plot(DIR_NAME)