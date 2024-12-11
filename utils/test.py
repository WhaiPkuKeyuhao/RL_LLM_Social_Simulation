import sys
sys.path.append(".")
sys.path.append("../")

import argparse
import pandas as pd
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from envs.tools import create_folder_if_not_exists
from utils.episode_data_process import episode_tocsv
import numpy as np
import matplotlib.dates as mdates

def convert_min_to_time(min):
    hour_part = min//60
    minute_part = min%60
    return f"{hour_part}:{minute_part}"

CN_TO_EN = {
    "在家里": "at home",
    "在路上": "on the way",
    "在工作地": "at working space",
    "在学校": "at school",
    "在小区商场": "at the shop",
    "在小区公共场所": "in the community",
    "睡觉": "sleep",
    "保持": "keep",
    "放松": "relax",
    "吃饭": "eat",
    "散步": "walk",
    "阅读": "read",
    "运动": "exercise",
    "工作": "work",
    "学习": "study",  # 在学校学习
    "购物": "buy",  # 在商场购物(商场认为在社区里面)
    "回家": "go home",
    "去工作地": "go to work",
    "去学校": "go to school",
    "去小区商场": "go to shop",
    "去小区公共场所": "go to community",
    "未成年学生": "Juvenile Student",
    "大学生": "College Student",
    "上班族": "Employed Adult",
    "自由职业": "Unemployed Adult",
    "退休": "Retired Senior"
}

DIR_NAME = "../data/train_9_26"
FILE_NAME = "data_episode_5_1.txt"
AGENT_NUMS = 256
WEEK_HOLIDAY = "周一工作日"
NAME = "episode_5"
RESIDENT_PATH = "../config/agentSimulate10000.json"



file_path = os.path.join(DIR_NAME, f"{FILE_NAME}")
with open(file_path, 'r') as f:
    lines = f.readlines()
fileDir = os.path.join(DIR_NAME, FILE_NAME)
create_folder_if_not_exists(fileDir)

# make df [person_ids,history_of_person]
person_ids = []
history_of_person = []
for i, x in enumerate(lines):
    if i % 2 == 0:
        person_ids.append(x.strip())
    else:
        history_of_person.append(json.loads(x.strip()))
df = pd.DataFrame(
    {'person_ids': pd.Series(person_ids)[-AGENT_NUMS:],
     'history_of_person': pd.Series(history_of_person)[-AGENT_NUMS:]})
df = df.reset_index(drop=True)
#print(df.head())
# person_ids, history_of_person


resident_list = pd.read_json(RESIDENT_PATH, encoding='gbk')
#print(resident_list.head())
resident_list['agent_id'] = resident_list.apply(lambda r: f'''person_{r["house_ue_id"]}_{r["id"]}''', 1)
#print(resident_list['agent_id'][:9])
life_stages = [ "未成年学生", "大学生", "上班族", "自由职业","退休"]

person_id_dic = {"未成年学生":[],
                 "大学生":[],
                 "上班族":[],
                 "自由职业":[],
                 "退休":[]}

#从resident_list中找到相符的人并且放到对应的人生阶段中

for person_id in df["person_ids"].tolist():
    for agent_id in resident_list['agent_id'].tolist():
        if person_id == agent_id:
            life_stage = resident_list[ resident_list["agent_id"] ==  agent_id]["人生阶段"].tolist()[0]
            person_id_dic[life_stage].append(person_id)


#在某一时刻time，,在人群life_stage中数有多少人在做action, 有多少人在location
# person_id_dic 记录不同人生阶段的人的id, df中的“history_of_person”记录了该agent的动作历史
def count_life_stage(time, life_stage,person_id_dic,df):
    #重置两个个字典，一个地点，一个动作
    location_dict = {
                    "at home": 0,
                    "on the way": 0,
                    "at working space": 0,
                    "at school": 0,
                    "at the shop": 0,
                    "in the community": 0
    }
    action_dict = {
        "sleep": 0,
        "keep": 0,
        "relax": 0,
        "eat": 0,
        "walk": 0,
        "read": 0,
        "exercise": 0,
        "work": 0,
        "study": 0,
        "buy": 0,
        "go home": 0,
        "go to work": 0,
        "go to school": 0,
        "go to shop": 0,
        "go to community": 0
    }
    #提取需要的id
    id_list = person_id_dic[life_stage]
    df_id = df["person_ids"].tolist()
    df_history = df["history_of_person"].tolist()
    #遍历df找id
    for id in id_list:
        for idx in range(len(df_id)):
            # 如果id对上了，找到对应的history里面找
            if id == df_id[idx]:
                history = df_history[idx]
                # 在这个history中找对应的时间点
                for record in history:
                    if record["start_time"]<= time and record["end_time"]>= time:
                        #找到时间点后，往location_dict和action_dict里面记录
                        action = record["action"]
                        location = record["location"]
                        action_dict[CN_TO_EN[action]] += 1
                        location_dict[CN_TO_EN[location]] += 1

    return action_dict, location_dict
for life_stage in life_stages:
    action_time_dict = {"sleep": [],
        "keep": [],
        "relax": [],
        "eat": [],
        "walk": [],
        "read": [],
        "exercise": [],
        "work": [],
        "study": [],
        "buy": [],
        "go home": [],
        "go to work": [],
        "go to school": [],
        "go to shop": [],
        "go to community": [],
        "time" : pd.date_range(start='2023-01-01', periods=24*6, freq='10T')}
    location_time_dict = {"at home": [],
                    "on the way": [],
                    "at working space": [],
                    "at school": [],
                    "at the shop": [],
                    "in the community": [],
                    "time" : pd.date_range(start='2023-01-01', periods=24*6, freq='10T')
                          }
    for time in range(0, 24 * 60, 10):
        action_dict, location_dict = count_life_stage(time, life_stage,person_id_dic,df)
        for key in action_dict.keys():
            action_time_dict[key].append(action_dict[key])
        for key in location_dict.keys():
            location_time_dict[key].append(location_dict[key])


    df_plot = pd.DataFrame(location_time_dict)

    # 设置Seaborn样式
    sns.set(style="whitegrid")

    # 绘制图表
    plt.figure(figsize=(12, 6))
    for feature in df_plot.columns.tolist():
        if feature != "time":
            sns.lineplot(x='time', y=feature, data=df_plot, label=feature, linewidth=2.5)

    # 设置x轴刻度，每隔2小时标记一次
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签，避免重叠

    # 将图例放置在图表的最右侧
    plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)

    # 设置标题和轴标签
    plt.title(f'People of {CN_TO_EN[life_stage]} in Each Location ')
    plt.ylabel('Number of People')

    # 显示图表
    plt.tight_layout()  # 调整布局以适应图例
    plt.savefig(f"{CN_TO_EN[life_stage]}_location_count")
    plt.close()

    df_plot = pd.DataFrame(action_time_dict)

    # 设置Seaborn样式
    sns.set(style="whitegrid")

    # 绘制图表
    plt.figure(figsize=(12, 6))
    for feature in df_plot.columns.tolist():
        if feature != "time":
            sns.lineplot(x='time', y=feature, data=df_plot, label=feature, linewidth=2.5)

    # 设置x轴刻度，每隔2小时标记一次
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签，避免重叠

    # 将图例放置在图表的最右侧
    plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)

    # 设置标题和轴标签
    plt.title(f'People of {CN_TO_EN[life_stage]} in Each Action ')
    plt.ylabel('Number of People')

    # 显示图表
    plt.tight_layout()  # 调整布局以适应图例
    plt.savefig(f"{CN_TO_EN[life_stage]}_action_count")
    plt.close()






