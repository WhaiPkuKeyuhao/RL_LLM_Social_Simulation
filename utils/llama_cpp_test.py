# coding: utf-8

import sys
sys.path.append(".")
sys.path.append("../")

import aiohttp
import asyncio
import pandas as pd
import re
from envs.tools import is_number



url0 = f'http://172.16.10.102:8000/v1/chat/completions'
url1 = f'http://172.16.10.102:8001/v1/chat/completions'
url2 = f'http://172.16.10.102:8002/v1/chat/completions'

###  测评大模型, 跟人类反馈结果进行对比 

test_prompt = '''这是一个计算机模拟的小区及小区居民,居民只有6种地理位置状态:家、路上、工作地、学校、小区商场、小区公共场所,小区居民只有15种动作:睡觉、保持、休息娱乐、吃饭、散步、阅读、运动、工作、学习、购物、回家、去工作地、去学校、去小区商场、去小区公共场所,周一到周五是工作日,上班族都得去工作,工作时间一般在8点到18点之间,学生都
得去上学,一般在7点到17点在学校学习,其他时间在工作地或者在学校都是不合理的,自由职业和退休人员一般都呆在家里,有时候去小区公共场所散步或运动,去小区商场买东西,晚上23点到早上6点间在社区活动不合理,模拟小区开始模拟时间是周一早上0点,所有居民都是睡觉状态,居民每天睡觉8小时左右,晚上23点到早上7点之间睡觉都合理,我们想在这个
小区中模拟出正常人的行为,需要你对模拟人的状态行为打分,以保证模拟人会越来越接近正常人类,许萍是一名正常39岁的模拟小区居民, 女性,她未婚, 受教育程度是大专, 目前是处于自由职业阶段,今天是2024年2月的一天, 周一工作日, 北京时间0点0分, 她在家里睡觉,在0点5分到0点10分, 她去小区公共场所,在0点10分到0点15分, 她回家,在0点15分到
0点20分, 她去小区公共场所,在0点20分到0点30分, 她在小区公共场所运动,现在时间是0点30分,她在小区公共场所运动,请回答她目前行为状态的合理程度,主要考虑时间、地点、行为以及历史行为,100为合理,0为不合理,50为不确定是否合理请结合上述让模拟趋近真实的意图,直接给出0、50、100数值就行,不要输出任何标点符号,不要说其他任何话,不用说明理由'''


async def generate_response(message):
    

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
                    'content': message,
                    'role': 'user'
                    },
            ],
            "temperature": 0,
            "seed": 0
    }


    result0 = await request_url(url0, headers, data)
    result1 = await request_url(url1, headers, data)
    #result2 = await request_url(url2, headers, data)

    result = (result0 + result1) / 2

    return result

async def request_url(url, headers, data):
    result = 50
    try:
        async with aiohttp.ClientSession() as session:
            score = "unknown"
            async with session.post(url, headers=headers, json=data) as response:
                    assert response.status == 200 
                    result = await response.json()
                    print(result)
                    score = result['choices'][0]['message']['content']

                    if not is_number(score):
                        match = re.search(r'\d+', score)
                        if match:
                            score = match.group()
                        else:
                            score = 50
            result = float(score)
    except Exception as e:
        print(e)
    return result

if __name__ == "__main__":
    #asyncio.run(generate_reponse(test_prompt))

    df = pd.read_excel("data/人类反馈智能体活动评分问卷_得分.xlsx", usecols=[0, 1, 2])
    human_scores = []
    for index, row in df.iterrows():
        score = row["score"]
        human_scores.append(float(score))

    result = 0   

    #asyncio.run(create_response())
    with open("data/prompt_test.txt", "r") as f:
        for idx, line in enumerate(f):
            #prompts.append(line)
            score = asyncio.run(generate_response(line))
            diff = abs(human_scores[idx] - float(score))
            result += diff
    print("result: {}".format(result))

    # diff = abs(human_scores[idx] - float(score))
    # 
    # result += diff
    # print("result: {}".format(result))