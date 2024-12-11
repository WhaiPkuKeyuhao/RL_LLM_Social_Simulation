import sys
sys.path.append(".")
sys.path.append("../")

import argparse
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from envs.tools import create_folder_if_not_exists
import matplotlib.dates as mdates
import numpy as np
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

count_location_defaultValue = {"at home": 0,
                                "on the way": 0,
                                "at working space": 0,
                                "at school": 0,
                                "at the shop": 0,
                                "in the community": 0}
count_action_defaultValue = {"sleep": 0,
                                #"keep": 0,
                                "relax": 0,
                                "eat": 0,
                                "walk": 0,
                                "read": 0,
                                "exercise": 0,
                                "work": 0,
                                "study": 0, 
                                "buy": 0,
                                # "go home": 0,
                                # "go to work": 0,
                                # "go to school": 0,
                                # "go to shop": 0,
                                # "go to community": 0
                                }

color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white',
          'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
cmap = colors.ListedColormap(color_list)


def episode_tocsv(fileDir, fileName, numAgent, name):
    f=open(os.path.join(fileDir, fileName),'r')
    lines=f.readlines()

    person_ids=[]
    history_of_person=[]
    for i,line in enumerate(lines):
        if i%2==0:
            person_ids.append(line.strip())
        else:
            history_of_person.append(json.loads(line.strip()))

    df=pd.DataFrame({'person_ids':pd.Series(person_ids)[-numAgent:],'history_of_person':pd.Series(history_of_person)[-numAgent:]})

    df=df.reset_index(drop=True)
    #从0:0开始 0分到24*60分，每5分钟统计一次
    count={}#玩手机人数count
    def function(r,minite):
        count.setdefault(f'{minite//60}:{minite%60:02d}',0)
        for d in r['history_of_person']:
            if d['start_time']<=minite and minite < d['end_time']:
                #print(d['location'])
                if (d['location']=='在家里' and d['action'] in ("放松", "阅读")) or \
                 (d['location'] in ('在小区商场', '在小区公共场所') and d['action'] in ("运动", "购物")):
                    count[f'{minite//60}:{minite%60:02d}']+=1

    for minite in range(0,24*60,5):
        for r in range(len(df)):#每行
            function(df.iloc[r],minite)
    
    count_paly={}        
    for k in count:
        count_paly[k]=[f'{count[k]/(numAgent)*100:.2f}%']
    count_paly=pd.DataFrame(count_paly)
    count_paly['type']=name

    origin_csv = os.path.join("data", "out.csv")

    target_dir = create_folder_if_not_exists(os.path.join(fileDir, name))
    target_csv = os.path.join(target_dir, f"out_{name}.csv")
    d=pd.read_csv(origin_csv,encoding='gbk')
    d=pd.concat((d,count_paly))
    d.to_csv(target_csv, index=False,encoding='gbk')


def plot(fileDir, type, name):
    fileDir = os.path.join(fileDir, name)
    csv_file = os.path.join(fileDir, f"out_{name}.csv")
    df = pd.read_csv(csv_file,encoding='gbk')
    d_true = df[df['type']==type]

    d_nihe = df[df['type']==name]
    
    d_true=d_true[d_true.columns[1:]]
    d_true1=[]
    for i,x in enumerate(d_true.iloc[0]):
        if x.split('%')[0]!='----':
            d_true1.append(float(x.split('%')[0])/100 )
        else:
            d_true1.append(d_true1[i-1])#用前热力值补空值
    
    d_nihe=d_nihe[d_nihe.columns[1:]]
    

    d_nihe1=[]
    for i,x in enumerate(d_nihe.iloc[0]):
        d_nihe1.append(float(x.split('%')[0])/100 )
    plt.rcParams ["font.family"] = "SimHei"
    d = pd.DataFrame({'time (minutes)':list(range(0,60*24,5)), 'value':d_true1, name:d_nihe1})
    f=sns.lineplot(x='time (minutes)', y='value' ,data=d,label=type)
    f=sns.lineplot(x='time (minutes)', y=name ,data=d,label=name)#x轴用分钟表示
    f.get_figure().savefig(os.path.join(fileDir, f'热力值拟合{name}.png'), dpi = 1000)


def plot_agentNum_in_location_action(fileDir, FILE_NAME, numAgent, name):#需要改一下count_location_defaultValue和count_action_defaultValue的值
    file_path= os.path.join(fileDir, f"{FILE_NAME}")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    fileDir = os.path.join(fileDir, name)
    person_ids = []
    history_of_person = []
    for i, x in enumerate(lines):
        if i % 2 == 0:
            person_ids.append(x.strip())
        else:
            history_of_person.append(json.loads(x.strip()))
    
    print("person_ids size: {} history_of_person size: {}".format(len(person_ids), len(history_of_person)))
    df = pd.DataFrame(
        {'person_ids': pd.Series(person_ids)[-numAgent:], 'history_of_person': pd.Series(history_of_person)[-numAgent:]})
    df = df.reset_index(drop=True)

    #print(history_of_person[4])
    

    # 从0:0开始 0分，24*60分 ，5分钟一统计，不同location state 和 不同action 对应的人数
    count_location = {}
    count_action = {}



    # colors =sns.color_palette()
    # colors.extend(sns.color_palette('Set2_r'))
    # colors[13]=sns.color_palette('Set2_r')[-1]

    def fun(r, minite):
        count_location.setdefault(minite, count_location_defaultValue.copy())
        count_action.setdefault(minite, count_action_defaultValue.copy())
        
        for d in r['history_of_person']:
            if d['start_time'] <= minite and minite <= d['end_time']:
                count_location[minite][CN_TO_EN[d['location']]] += 1
                if CN_TO_EN[d['action']] in count_action_defaultValue.keys():
                    count_action[minite][CN_TO_EN[d['action']]] += 1

    
    #print(df.iloc[0]['history_of_person'])

    for minite in range(0, 24 * 60, 10):
        for r in range(len(df)):
            fun(df.iloc[r], minite)

    print(count_location[100])
    print("-----------")
    print(count_action[100])

    #plt.rcParams["font.family"] = "SimHei"
    plt.figure()
    data = pd.read_json(json.dumps(count_location)).T#处理count_location
    colors = plt.cm.tab10(np.linspace(0,1,6))
    fig_location = data.plot(color = colors)
    plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    plt.xlabel('time/seconds')
    plt.ylabel('agent number each location')
    fig_location.figure.savefig(os.path.join(fileDir, f'count_location-{name}.png'), dpi=1000, bbox_inches='tight')

    plt.figure()
    data = pd.read_json(json.dumps(count_action)).T#处理count_action
    colors = plt.cm.tab20(np.linspace(0,1,15))
    fig_action = data.plot(color = colors)
    plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    plt.xlabel('time/seconds')
    plt.ylabel('agent number each action')
    fig_action.figure.savefig(os.path.join(fileDir, f'count_action-{name}.png'), dpi=1000, bbox_inches='tight')



                                              #txt
def plot_agentNum_by_life_stage(dir_name, FILE_NAME, numAgent, resident_path, name):
    file_path = os.path.join(dir_name, f"{FILE_NAME}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    fileDir = os.path.join(DIR_NAME, name)
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
        {'person_ids': pd.Series(person_ids)[-numAgent:],
         'history_of_person': pd.Series(history_of_person)[-numAgent:]})
    df = df.reset_index(drop=True)
    # print(df.head())
    # person_ids, history_of_person

    resident_list = pd.read_json(resident_path, encoding='gbk')
    # print(resident_list.head())
    resident_list['agent_id'] = resident_list.apply(lambda r: f'''person_{r["house_ue_id"]}_{r["id"]}''', 1)
    # print(resident_list['agent_id'][:9])
    life_stages = ["未成年学生", "大学生", "上班族", "自由职业", "退休"]

    person_id_dic = {"未成年学生": [],
                     "大学生": [],
                     "上班族": [],
                     "自由职业": [],
                     "退休": []}

    # 从resident_list中找到相符的人并且放到对应的人生阶段中

    for person_id in df["person_ids"].tolist():
        for agent_id in resident_list['agent_id'].tolist():
            if person_id == agent_id:
                life_stage = resident_list[resident_list["agent_id"] == agent_id]["人生阶段"].tolist()[0]
                person_id_dic[life_stage].append(person_id)

    # 在某一时刻time，,在人群life_stage中数有多少人在做action, 有多少人在location
    # person_id_dic 记录不同人生阶段的人的id, df中的“history_of_person”记录了该agent的动作历史
    def count_life_stage(time, life_stage, person_id_dic, df):
        # 重置两个个字典，一个地点，一个动作
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
        # 提取需要的id
        id_list = person_id_dic[life_stage]
        df_id = df["person_ids"].tolist()
        df_history = df["history_of_person"].tolist()
        # 遍历df找id
        for id in id_list:
            for idx in range(len(df_id)):
                # 如果id对上了，找到对应的history里面找
                if id == df_id[idx]:
                    history = df_history[idx]
                    # 在这个history中找对应的时间点
                    for record in history:
                        if record["start_time"] <= time and record["end_time"] >= time:
                            # 找到时间点后，往location_dict和action_dict里面记录
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
                            "time": pd.date_range(start='2023-01-01', periods=24 * 6, freq='10T')}
        location_time_dict = {"at home": [],
                              "on the way": [],
                              "at working space": [],
                              "at school": [],
                              "at the shop": [],
                              "in the community": [],
                              "time": pd.date_range(start='2023-01-01', periods=24 * 6, freq='10T')
                              }
        for time in range(0, 24 * 60, 10):
            action_dict, location_dict = count_life_stage(time, life_stage, person_id_dic, df)
            for key in action_dict.keys():
                action_time_dict[key].append(action_dict[key])
            for key in location_dict.keys():
                location_time_dict[key].append(location_dict[key])

        df_plot = pd.DataFrame(location_time_dict)
        df_plot.to_csv("location_count_against_time.csv")
        # 设置Seaborn样式
        sns.set(style="whitegrid")
        colors = plt.cm.tab10(np.linspace(0,1,6))
        # 绘制图表
        plt.figure(figsize=(25, 8))
        for i, feature in enumerate(df_plot.columns.tolist()):
            if feature != "time":
                sns.lineplot(x='time', y=feature, data=df_plot, label=feature, linewidth=2.5, color=colors[i])

        # 设置x轴刻度，每隔2小时标记一次
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()  # 自动旋转日期标签，避免重叠
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.gca().tick_params(axis='y', labelsize=25)

        # 将图例放置在图表的最右侧
        plt.legend(bbox_to_anchor=(1.005, 0), loc='lower left', borderaxespad=0., fontsize=15)

        # 设置标题和轴标签
        plt.title(f'People of {CN_TO_EN[life_stage]} in Each Location ',fontsize = 35)
        plt.ylabel('Number of People',fontsize=28)
        plt.xlabel('Time', fontsize=28)

        # 显示图表
        plt.tight_layout()  # 调整布局以适应图例
        plt.savefig(fileDir+f"/{CN_TO_EN[life_stage]}_location_count")
        plt.close()

        df_plot = pd.DataFrame(action_time_dict)
        df_plot.to_csv("action_count_against_time.csv")
        # 设置Seaborn样式
        sns.set(style="whitegrid")
        colors = plt.cm.tab20(np.linspace(0,1,15))
        # 绘制图表
        plt.figure(figsize=(25, 8))
        for i, feature in enumerate(df_plot.columns.tolist()):
            if feature != "time":
                sns.lineplot(x='time', y=feature, data=df_plot, label=feature, linewidth=2.5,color = colors[i])

        # 设置x轴刻度，每隔2小时标记一次
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()  # 自动旋转日期标签，避免重叠
        plt.gca().tick_params(axis='x', labelsize=25)
        plt.gca().tick_params(axis='y', labelsize=25)
        # 将图例放置在图表的最右侧
        plt.legend(bbox_to_anchor=(1.005, 0), loc='lower left', borderaxespad=0.,fontsize = 15)

        # 设置标题和轴标签
        plt.title(f'People of {CN_TO_EN[life_stage]} in Each Action ',fontsize = 35)
        plt.ylabel('Number of People',fontsize=28)
        plt.xlabel('Time', fontsize=28)

        # 显示图表
        plt.tight_layout()  # 调整布局以适应图例
        plt.savefig(fileDir+f"/{CN_TO_EN[life_stage]}_action_count")
        plt.close()




if __name__=='__main__':

    DIR_NAME = "data\\train_0929f"
    FILE_NAME = "data_episode_1_7.txt"
    AGENT_NUMS = 256
    WEEK_HOLIDAY = "周日节假日"
    NAME = "episode_1"
    RESIDENT_PATH = "config/agentSimulate10000.json"

    episode_tocsv(DIR_NAME, FILE_NAME, AGENT_NUMS, NAME)
    plot(DIR_NAME, WEEK_HOLIDAY, NAME)
    plot_agentNum_in_location_action(DIR_NAME, FILE_NAME, AGENT_NUMS, NAME)
    # plot_agentNum_different_age_state(DIR_NAME, FILE_NAME, AGENT_NUMS, RESIDENT_PATH, NAME)
