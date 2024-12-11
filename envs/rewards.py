# coding: utf-8

import sys
sys.path.append(".")
sys.path.append("../")

import os
import re

import pandas as pd
from datetime import datetime, timedelta
import aiohttp
import random
from tools import is_number

now = datetime.now()


cur_dir = os.path.dirname(os.path.abspath(__file__))
community_location_heat_df = pd.read_csv(os.path.join(cur_dir, "../config/community_location_heat.csv"))
community_location_heat_data = community_location_heat_df.to_dict(orient='index')

WEEK_HOLIDAY_TYPE_DICT = {
    "周一工作日": 0,
    "周一节假日": 1,
    "周二工作日": 2,
    "周二节假日": 3,
    "周三工作日": 4,
    "周三节假日": 5,
    "周四工作日": 6,
    "周四节假日": 7,
    "周五工作日": 8,
    "周五节假日": 9,
    "周六节假日": 11,
    "周日工作日": 12,
    "周日节假日": 13,
}
     

#session=requests.Session()

async def request_url(url, headers, data):
    result = 50
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                    assert response.status == 200 
                    result = await response.json()
                    
                    score = result['choices'][0]['message']['content']

                    if not is_number(score):
                        match = re.search(r'\d+', score)
                        if match:
                            score = match.group()
                        else:
                            score = 50
                    #print("{} {}".format(result, score))
            result = float(score)
    except Exception as e:
        print(e)
    return result


async def get_reward(idx, message, history=[]):
    assert(len(history) > 0)

    ### statistics reward
    statistics_target = message["statistics_target"]
    statistics_actual = message["statistics_actual"]
    in_community_and_use_phone = message["in_community_and_use_phone"]

    statistics_reward = 0
    if statistics_target - statistics_actual > 0.1:
        if in_community_and_use_phone == 1:
            statistics_reward = 0.5
        else:
            statistics_reward = -0.5
    
    if statistics_target - statistics_actual < -0.1:
        if in_community_and_use_phone == 1:
            statistics_reward = -0.5
        else:
            statistics_reward = 0.5

    url0 = f'http://172.16.10.102:8000/v1/chat/completions'
    url1 = f'http://172.16.10.102:8001/v1/chat/completions'


    pronouns = "他" if message["gender"] == "男性" else "她"
    name = message["name"]
    
    _background = "这是一个计算机模拟的小区及小区居民,居民只有6种地理位置状态:家、路上、工作地、学校、小区商场、小区公共场所,"
    _background += "小区居民只有15种动作:睡觉、保持、休息娱乐、吃饭、散步、阅读、运动、工作、学习、购物、回家、去工作地、去学校、去小区商场、去小区公共场所,"
    _background += "周一到周五是工作日,上班族都得去工作,工作时间一般在8点到18点之间,学生都得去上学,一般在7点到17点在学校学习,其他时间在工作地或者在学校都是不合理的,"
    _background += "自由职业和退休人员一般都呆在家里,有时候去小区公共场所散步或运动,去小区商场买东西,晚上23点到早上6点间在社区活动不合理,"
    _background += "模拟小区开始模拟时间是周一早上0点,所有居民都是睡觉状态,居民每天睡觉8小时左右,晚上23点到早上7点之间睡觉合理,其他时间睡觉不合理"
    _background += "我们想在这个小区中模拟出正常人的行为,需要你对模拟人的状态行为打分,以保证模拟人会越来越接近正常人类,"
    _background += "我们对话中提到的所有时间都将基于24小时制,不是12小时制,这非常重要！"
    _content = f"{name}是一名正常{message['age']}岁的模拟小区居民, {message['gender']}," 
    _content += f"""{pronouns}{message["marriage_status"]},受教育程度是{message["education_level"]},目前是处于{message["life_stage"]}阶段,"""
    if message["life_stage"] in ("上班族", "未成年学生"):
        _content += f"""{pronouns}工作日每天都习惯在{message["action_time"]}{message["probable_action"]},{message["go_home_time"]}的时候回家。"""

    for i, d in enumerate(history):
        day_of_week, start_time, end_time, location, action, target_location = d.values()

        status_action = f"{location}{action}"
        if location == "在路上":
            if target_location != "":
                status_action = "在去" + target_location.split()[0][1:] + "的路上"
            else:
                status_action = location
        elif action in ("回家", "去工作地", "去学校", "去小区商场", "去小区公共场所"):
            status_action = action

        if i==0:
            _content += f"今天是{day_of_week}, 北京时间{start_time//60}点{start_time%60}分, {pronouns}{location},"
        else:
            _content += f"在{start_time//60}点{start_time % 60}分到{end_time//60}点{end_time%60}分, {pronouns}{status_action},"

    status_action = f'''{message["location"]}{message["action"]}'''
    if message["location"] == "在路上":
        if message["target_location"] != "":
            #print(message["target_location"])
            status_action = "在去" + message["target_location"].split()[0][1:] + "的路上"
        else:
            status_action = message["location"]
    elif message["action"] in ("回家", "去工作地", "去学校", "去小区商场", "去小区公共场所"):
        status_action = message["action"]

    _content += f'''现在时间是{message["hours"]}点{message["minutes"]}分,{pronouns}{status_action},'''

    if idx % 40 == 0:
        print(_content)


    _content += f'''请回答{pronouns}现在行为的合理程度,主要考虑在当前时间做出的行为是否合理,100为合理,0为不合理,50为不确定是否合理,'''
    _content += "请结合上述让模拟趋近真实的意图,直接给出0到100之间数值就行,不要说其他任何话,不用说明理由"
    _content = _background + _content



    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
            'messages': [
                    {
                    'content': "只需要回答分数,不要回答其他",
                    'role': 'system'
                    },
                    {
                    'content': _content,
                    'role': 'user'
                    },
            ],
            "temperature": 0,
            "seed": 0
    }

    result0 = await request_url(url0, headers, data)
    result1 = await request_url(url1, headers, data)
    score = (result0 + result1) / 2
    llm_reward = (float(score) - 60) / 100

    if llm_reward > 0.3:
        statistics_reward = abs(statistics_reward)

    return llm_reward, statistics_reward


def adjust_time(time_str, delta_minutes):
    # 将时间字符串转换为datetime对象
    time = datetime.strptime(time_str, "%H:%M")
    time += timedelta(minutes=delta_minutes)
    hour, minute = map(int, time.strftime("%H-%M").split("-"))

    def map_to_05(num):
        return int(str(num)[:-1] + ('0' if num % 10 < 5 else '5'))

    minute = map_to_05(minute)    
    return "{:02d}:{:02d}".format(hour, minute)

def get_statistics_reward(message):

    # 社区人口热力统计, 统计在家且在使用手机人的比例
    if message["week_holiday"] in WEEK_HOLIDAY_TYPE_DICT.keys():
        data = community_location_heat_data[WEEK_HOLIDAY_TYPE_DICT[message["week_holiday"]]]
        t = adjust_time(message["time"], 0)

        p = float(data[t].strip("%")) / 100
        q = message["actual"]
        return -abs((p - q))
    return 0
        
def get_statistics_target(message):
    if message["week_holiday"] in WEEK_HOLIDAY_TYPE_DICT.keys():
        data = community_location_heat_data[WEEK_HOLIDAY_TYPE_DICT[message["week_holiday"]]]
        t = adjust_time(message["time"], 0)
        p = float(data[t].strip("%")) / 100
        return p
    else:
        return 0
    
def get_next_hours_statistics_target(message, action_interval):
    result = []
    if message["week_holiday"] in WEEK_HOLIDAY_TYPE_DICT.keys():
        data = community_location_heat_data[WEEK_HOLIDAY_TYPE_DICT[message["week_holiday"]]]
        
        for i in range(1, 6, 1):
            t = adjust_time(message["time"], i * action_interval)
            p = float(data[t].strip("%")) / 100
            result.append(p)
        return result
    else:
        return result


if __name__ == "__main__":
    result = get_statistics_reward({
        "week_holiday" : "周二工作日",
        "time": "01:13",
        "actual": 0.1,
    })
    #print(result)

    result_ = get_statistics_target({
        "week_holiday" : "周五工作日",
        "time": "01:13",
        "actual": 0.1,
    })

    result_ = get_next_hours_statistics_target({
        "week_holiday" : "周一工作日",
        "time": "23:43",
        "actual": 0.1,
    })
    print(result_)