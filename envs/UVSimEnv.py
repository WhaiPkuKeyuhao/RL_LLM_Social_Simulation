import sys
from random import randint

from numpy.random import random_integers

sys.path.append(".")
sys.path.append("../")

import os
from airsim import CarClient
import json
import random
import numpy as np
import time
from datetime import datetime
from shapely import Point, Polygon
from collections import Counter
import configparser
from ray import rllib
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
#from llama3_1 import generate_reponse
import onnxruntime
import asyncio

from tools import seconds_to_dhms, create_folder_if_not_exists
from rewards import get_reward, get_statistics_target, get_next_hours_statistics_target

cur_dir = os.path.dirname(os.path.abspath(__file__))

config = configparser.ConfigParser()
config.read(os.path.join(cur_dir, f"../config/config.ini"), encoding="utf-8")
AGENT_ACTION_DICT = eval(config["resident_agent"]["agent_action_dict"])
FACILITY_LIST = eval(config["resident_agent"]["facility_list"])
LOCATION_DICT = eval(config["resident_agent"]["location_dict"])

LOCATION_ACTION_MASK = eval(config["resident_agent"]["location_action_mask"])
LIFE_STAGE_ACTION_MASK = eval(config["resident_agent"]["life_stage_action_mask"])
# add another schedule MASK
SCHEDULE_ACTION_MASK = eval(config["resident_agent"]["schedule_action_mask"])

AGE_TYPE_DICT = eval(config["resident_agent"]["age_type_dict"])
AGE_STATE_DICT = eval(config["resident_agent"]["age_state_dict"])
LIFE_STAGE_DICT = eval(config["resident_agent"]["life_stage_dict"])

AGENT_ACTION_DICT_CN = eval(config["resident_agent"]["agent_action_dict_cn"])
LOCATION_DICT_CN = eval(config["resident_agent"]["location_dict_cn"])
TRAINING_DAYS = eval(config["resident_agent"]["training_days"])

DATA_DIR = "data"

#WEEK_HOLIDAY = "周一工作日"

ACTION_SIZE = len(list(AGENT_ACTION_DICT.keys()))

LOCATION_STATE = np.eye(len(LOCATION_DICT.keys()))
ACTION_STATE = np.eye(len(AGENT_ACTION_DICT.keys()))
HOURS_STATE = np.eye(24)
AGE_STATE = np.eye(len(AGE_STATE_DICT.keys()))

OBS_SIZE = 69


class SimEnv(rllib.MultiAgentEnv):
    def __init__(self, speed, start_time, total_hours, total_agent_num, action_duration_size, action_interval, task_name, is_training=True, ip_address="172.25.224.1", port=41451):
        with open(os.path.join(cur_dir, f"../config/agentSimulate10000_0926.json"), "r", encoding="gbk") as f:
            self.resident_list = json.load(f)

        self.ip_address = ip_address
        self.port = port

        self.client = CarClient(ip=self.ip_address, port=self.port)
        self.client.confirmConnection()
        

        self.speed = speed
        self.start_ts = start_time
        self.total_hours = total_hours
        self.total_agent_num = total_agent_num
        self.action_duration_size = action_duration_size
        self.action_interval = action_interval
        self.task_name = task_name
        self.is_training = is_training

        current_date_and_time = datetime.now()
        self.timeofday = (str(current_date_and_time.year)+ "-" + str(current_date_and_time.month) + "-" + str(current_date_and_time.day))
        self.action_space = Discrete(ACTION_SIZE, action_duration_size)
        self.observation_space = Dict({
            "action_mask": Box(low=0, high=1, shape=(ACTION_SIZE + self.action_duration_size,), dtype=np.float32),
            "observations": Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        })

        self.episode = 0
        self.m_loop = asyncio.get_event_loop()
        self.WEEK_HOLIDAY = "周一工作日"
        self.timeHour = 0

        random.seed(42)

    def reset(self, *, seed=None, options=None):
        self.close()

        ## todo: 保存当前episode智能体状态动作信息
        
        self.agent_list = []
        self.building_set = set()
        self.home_pos_dict = {}
        self.persons_dict = {}
        self.state_dict = {}
        self.agent_history_for_score = {}
        self.building_polygon = {}
        self.facility_dict = {}
        self.m_step = 0

        #初始化 average_llm_reward_list 和 llm_reward_list
        self.average_llm_rewar_list = []
        self.llm_reward_list = []


        persons = []
        if self.episode == 0: # 暂时先固定住人员配置
            random.shuffle(self.resident_list)

        
        count_life_stage_type = {"上班族": 0, 
                                 "退休": 0,
                                "自由职业": 0,
                                "大学生": 0,
                                "未成年学生": 0,}
        
        new_resident_list = []
        for i, resident in enumerate(self.resident_list):
            # 适量减少上班族、学生人数
            if resident["人生阶段"] in ("上班族", "未成年学生") and random.randint(0, 100) < 40:
                continue
            new_resident_list.append(resident)

        resident_list = new_resident_list[:self.total_agent_num]

        for i, resident in enumerate(resident_list):
            house_name = resident["house_ue_id"]
            home_pos = self.client.getObjectPose(house_name)

            pos = home_pos.position
            self.building_set.add("_".join(house_name.split("_")[:-2]) + "_wd")

            resident_id = resident["id"]
            sex = resident["SEX"]
            gender_en = "male" if sex == "男性" else "female"
            age_type = resident["age_type"]

            age_state = age_type


            age = resident["AGE"]
            
            IDENTITY_ID = resident["IDENTITY_ID"]

            agent_id = f"person_{house_name}_{resident_id}"

            person = {
                "id": agent_id,
                "type": "human",
                "position": {"x": pos.x_val, "y": pos.y_val, "z": pos.z_val},
                "direction": {"yaw": random.randint(0, 360), "pitch": 0, "roll": 0},
                "identity_id": IDENTITY_ID,
                "age_type": age_type,
                "age": age,
                "gender": sex,
                "gender_en": gender_en,
                "name": resident["姓名"],
                "occupation": resident["职业"]
            }

            self.home_pos_dict["_".join(house_name.split("_")[1:-1])] = home_pos
            self.agent_list.append(agent_id)

            # 将对应的预期行为， 动作时间， 回家时间加进去
            self.persons_dict[agent_id] = {"name": resident["姓名"], "age": age, "gender": sex, "age_type": age_type,
                                                "age_state": age_state, "occupation": resident["职业"], "marriage_status": resident["婚姻状况"],
                                                "education_level": resident["受教育程度"], "life_stage": resident["人生阶段"],
                                                "car_owner": resident["是否有车"], "commuting_distance": resident["通勤距离"], "commuting_method": resident["通勤方式"],
                                                "probable_action":resident["预期行为"],"action_time":resident["行为时间"],"go_home_time":resident["回家时间"]}
            persons.append(person)

            count_life_stage_type[resident["人生阶段"]] += 1

        print("count_life_stage_type: {}".format(count_life_stage_type))


        self.client.addAgents(json.dumps(persons), "Human")
        print("add {} agents ok!".format(len(persons)))

        for i, agent_id in enumerate(self.agent_list):
            home_pos = self.client.getObjectPose(agent_id)
            pos = home_pos.position

            if pos.x_val ** 2 + pos.y_val ** 2 + pos.z_val ** 2 < 0.1:
                del self.agent_list[i]

            self.state_dict[agent_id] = {}
            self.state_dict[agent_id]["home_pos"] = home_pos
            self.state_dict[agent_id]["pos"] = home_pos
            self.state_dict[agent_id]["location_state"] = 0
            self.state_dict[agent_id]["action"] = AGENT_ACTION_DICT[1]
            self.state_dict[agent_id]["action_id"] = 1
            # add a new key to represent the scheduled action that will be done as soon as the agent is not on the way
            # initiate it as -1
            self.state_dict[agent_id]["scheduled_action"] = -1

            # agent历史用一个字典的5元组表示,初始时设置，时间用分钟存储
            # 将agent_history_for_sore 改为一个字典，每一天对应一个动作线
            self.agent_history_for_score[agent_id] = {}
            for k, v in TRAINING_DAYS.items():
                self.agent_history_for_score[agent_id][k] = [{"day_of_week": v, "start_time": 0 * 60, "end_time": 0 * 60,
                                                              "location":LOCATION_DICT_CN[0], "action": AGENT_ACTION_DICT_CN[1],
                                                              "target_location": LOCATION_DICT_CN[0]},]
        for building_id in self.building_set:
            self.building_polygon[building_id] = self.client.getObjectPolygon(building_id)

        for facility_name in FACILITY_LIST:
            self.facility_dict[facility_name] = {}
            facility_list = eval(self.client.getObjectsByTag(facility_name))
            for facility in facility_list:
                self.facility_dict[facility_name][facility] = self.client.getObjectPose(facility)
        #self.client.showAgentInfoBoard(False)
        self.ts = time.time()
        self.m_loop.run_until_complete(self.parse_obs())
        self.timeHour = 0
        print("-------------reset ok------------------")
        self.episode += 1
        return self.obs, self.infos
        

    def step(self, action_dict):
        print("-------------step {}------------------".format(self.m_step))
        self.m_step += 1
        show_date = self.timeofday + " " + f"{self.hours}:{self.minutes}:{self.seconds}"
        self.client.setTime(self.timeHour * 100, True, celestial_clock_speed=self.speed, start_datetime=show_date)

        for i, agent_id in enumerate(self.agent_list):
            #self.agent_action(agent_id, action_dict[agent_id][0], action_dict[agent_id][1])
            self.agent_action(agent_id, action_dict[agent_id], 0)
            self.client.setAgentLabelVisible(agent_id, True)
        
        self.update_agent_location_state()
        #self.show_building_statistic_info()
        self.m_loop.run_until_complete(self.parse_obs())
        #得到一个step中的llm_reward_list

        # 算出llm_reward_list 平均数
        if len(self.llm_reward_list)!=0:
            average = sum(self.llm_reward_list)/len(self.llm_reward_list)
        else:
            average = 0
        # 将新平均数放进average_llm_reward_list
        self.average_llm_rewar_list.append(average)
        # 重置 llm_reward_list
        self.llm_reward_list = []
        # 如果 m_step被144整除（一天走完了）把 average_llm_reward_list 写进 llm_reward_{episode}文件中
        if self.m_step % 144 == 0:
            data_dir = create_folder_if_not_exists(os.path.join(DATA_DIR, self.task_name))
            day_of_week = self.day_of_week-1
            if day_of_week == 0:
                day_of_week = 7


            file_name = f"llm_reward_episode_{self.episode}.txt"
            with open(os.path.join(data_dir, file_name), 'a+') as f:
                f.write(TRAINING_DAYS[day_of_week] + '\n')
                f.write(json.dumps(self.average_llm_rewar_list) + '\n')
            # 再重置average_llm_reward_list
            self.average_llm_rewar_list = []

            file_name_history = f"data_episode_{self.episode}_{day_of_week}.txt"
            with open(os.path.join(data_dir, file_name_history),'a+') as f:
                for _, agent_id in enumerate(self.agent_list):
                    f.write(agent_id +'\n')
                    f.write(json.dumps(self.agent_history_for_score[agent_id][day_of_week]) +'\n')

        return self.obs, self.rewards, self.terminateds, self.truncateds, self.infos
    

    async def parse_obs(self):

        self.timeHour = self.start_ts + ((self.action_interval * 60 * self.m_step) / 3600)

        self.day_of_week, self.hours, self.minutes, self.seconds = seconds_to_dhms(int(self.timeHour * 3600))

        self.day_of_week = self.day_of_week % 7
        if self.day_of_week == 0:
            self.day_of_week = 7
        obs, rewards, infos, terminateds, truncateds  = {}, {}, {}, {}, {}

        if self.timeHour < self.start_ts + self.total_hours:
            for _, agent_id in enumerate(self.agent_list):
                terminateds[agent_id] = False
                terminateds[agent_id] = False

            ## 统计数据差分reward
            in_community_and_use_phone_num = 0
            for _, agent_id in enumerate(self.agent_list):
                # 在家且且不在睡觉
                if (self.state_dict[agent_id]["location_state"] == 0 and self.state_dict[agent_id]["action_id"] in (2, 5)) or \
                    (self.state_dict[agent_id]["location_state"] in (4, 5) and self.state_dict[agent_id]["action_id"] in (6, 9)):
                    in_community_and_use_phone_num += 1
                    self.state_dict[agent_id]["in_community_and_use_phone"] = 1
                else:
                    self.state_dict[agent_id]["in_community_and_use_phone"] = 0

            self.WEEK_HOLIDAY = TRAINING_DAYS[self.day_of_week]

            statistics_target = get_statistics_target({"week_holiday": self.WEEK_HOLIDAY, "time": "{:02d}:{:02d}".format(self.hours, self.minutes)})
            statistics_actual = float(in_community_and_use_phone_num) / self.total_agent_num
            statistics_next_hours_target = get_next_hours_statistics_target({"week_holiday": self.WEEK_HOLIDAY, "time": "{:02d}:{:02d}".format(self.hours, self.minutes)}, self.action_interval)

            statistics_result = -abs(statistics_target - statistics_actual)
            print("in_community_and_use_phone_num: {}".format(in_community_and_use_phone_num))
            
            for idx, agent_id in enumerate(self.agent_list):
                week_feature = np.eye(7)[self.day_of_week - 1]                                                                                                                                       # 7
                weekend_feature = np.eye(2)[0 if "工作日" in self.WEEK_HOLIDAY else 1]                                                                                                                # 2
                location_feature = LOCATION_STATE[self.state_dict[agent_id]["location_state"]]                                                                                                       # 6
                last_location_featrue = LOCATION_STATE[self.state_dict[agent_id]["last_location_state"]] if "last_location_state" in self.state_dict[agent_id].keys() else LOCATION_STATE[0]         # 6
                action_feature = ACTION_STATE[self.state_dict[agent_id]["action_id"]]                                                                                                                # 15
                last_action_feature = ACTION_STATE[self.state_dict[agent_id]["last_action_id"]] if "last_action_id" in self.state_dict[agent_id].keys() else ACTION_STATE[0]                         # 15
                hours_feature = np.array([self.hours / 24])                                                                                                                                          # 24
                minutes_feature = np.array([self.minutes / 60])                                                                                                                                      # 1                                                                                  # 2
                life_stage_feature = np.eye(len(LIFE_STAGE_DICT.keys()))[LIFE_STAGE_DICT[self.persons_dict[agent_id]["life_stage"]]]                                                                  # 5

                last_statistics_result = self.state_dict[agent_id]["statistics_result"] if "statistics_result" in self.state_dict[agent_id].keys() else 0
                statistics_feature = np.array([statistics_actual, statistics_target, statistics_target - statistics_actual, \
                                                                statistics_result, last_statistics_result, self.state_dict[agent_id]["in_community_and_use_phone"]])                                  # 6
                statistics_next_hours_target_feature = np.array(statistics_next_hours_target)                                                                                                         # 5

                features = np.concatenate((week_feature, weekend_feature, location_feature, last_location_featrue, action_feature, last_action_feature, hours_feature, minutes_feature, #age_feature, gender_feature, marriage_feature, 
                                            life_stage_feature, statistics_feature, statistics_next_hours_target_feature))
                


                action_masks = self.get_action_masks(agent_id, idx)


                obs[agent_id] = {
                                "observations": features, 
                                "action_mask": np.concatenate((action_masks,
                                                                ))}
                
                self.state_dict[agent_id]["last_location_state"] = self.state_dict[agent_id]["location_state"]
                self.state_dict[agent_id]["last_action_id"] = self.state_dict[agent_id]["action_id"]
                self.state_dict[agent_id]["statistics_result"] = statistics_result

                if self.m_step > 0:
                    message = {"name": self.persons_dict[agent_id]["name"],
                            "age": self.persons_dict[agent_id]["age"],
                            "gender": self.persons_dict[agent_id]["gender"],
                            "location": LOCATION_DICT_CN[self.state_dict[agent_id]["location_state"]],
                            "action": AGENT_ACTION_DICT_CN[self.state_dict[agent_id]["action_id"]],
                            "day_of_week":self.WEEK_HOLIDAY,
                            "hours": self.hours,
                            "minutes": self.minutes,
                            "occupation": self.persons_dict[agent_id]["occupation"],
                            "marriage_status": self.persons_dict[agent_id]["marriage_status"],
                            "education_level": self.persons_dict[agent_id]["education_level"],
                            "life_stage": self.persons_dict[agent_id]["life_stage"],
                            "car_owner": self.persons_dict[agent_id]["car_owner"],
                            "commuting_distance": self.persons_dict[agent_id]["commuting_distance"],
                            "commuting_method": self.persons_dict[agent_id]["commuting_method"],
                            "target_location": LOCATION_DICT_CN[self.state_dict[agent_id]["target_location"]] if "target_location" in self.state_dict[agent_id].keys() else "",
                            "statistics_target": statistics_target,
                            "statistics_actual": statistics_actual,
                            "in_community_and_use_phone": self.state_dict[agent_id]["in_community_and_use_phone"],
                            "probable_action": self.persons_dict[agent_id]["probable_action"],
                            "action_time": self.persons_dict[agent_id]["action_time"],
                            "go_home_time": self.persons_dict[agent_id]["go_home_time"],
                            "in_schedule": self.state_dict[agent_id]["in_schedule"],
                            }

                    
                    if self.agent_history_for_score[agent_id][self.day_of_week][-1]["action"] == message["action"]:   # 此次动作跟上次动作相同
                        self.agent_history_for_score[agent_id][self.day_of_week][-1]['end_time'] = message["hours"] * 60 + message["minutes"]
                        llm_reward, statistics_reward = await get_reward(idx, message, self.agent_history_for_score[agent_id][self.day_of_week]) if self.is_training else (0, 0)
                    else:                                                                           # 此次动作跟上次不同
                        t = message["hours"] * 60 + message["minutes"]                     
                        self.agent_history_for_score[agent_id][self.day_of_week][-1]['end_time'] = message["hours"] * 60 + message["minutes"]
                        self.agent_history_for_score[agent_id][self.day_of_week].append({'day_of_week':self.WEEK_HOLIDAY,'start_time':t,'end_time': t, 'location':message['location'], "action":message['action'], "target_location":message['target_location']})  #end_time暂时设为相同, 放在-1下标，此事件暂时不用
                        llm_reward, statistics_reward = await get_reward(idx, message, self.agent_history_for_score[agent_id][self.day_of_week][:-1]) if self.is_training else (0, 0)

                    rewards[agent_id] = llm_reward + statistics_reward
                    #将llm_reward 放进llm_reward_list
                    self.llm_reward_list.append(llm_reward)

                    if not self.is_training:
                        time.sleep(0.01)

                    if idx % 40 == 0:
                        print("agent {} - {}岁 - {} ---- day_of_week:{}, hours {} minutes {} location {} action: {} target: {:.2f} actual: {:.2f} statistics_reward: {:.4f} llm_rewards:{} ".format(
                                idx, message["age"], message["life_stage"], self.WEEK_HOLIDAY,self.hours, self.minutes, LOCATION_DICT[self.state_dict[agent_id]["location_state"]], self.state_dict[agent_id]["action"],
                                statistics_target, statistics_actual, statistics_reward, llm_reward,))
            terminateds["__all__"] = False
            truncateds["__all__"] = False
        else:
            for _, agent_id in enumerate(self.agent_list):
                terminateds[agent_id] = True
                terminateds[agent_id] = True
            terminateds["__all__"] = True
            truncateds["__all__"] = True

            print('episode:', self.episode)

        self.obs = obs
        self.infos = infos
        self.rewards = rewards
        self.terminateds = terminateds
        self.truncateds = truncateds


    def update_agent_location_state(self):
        for i, agent_id in enumerate(self.agent_list):
            if self.state_dict[agent_id]["location_state"] == 1:
                des_reach = self.client.isMoveCompleted(agent_id)
                if des_reach:
                    # 跑到即算到达
                    self.state_dict[agent_id]["location_state"] = self.state_dict[agent_id]["target_location"]



    def agent_action(self, agent_id, action_id):
        agent_action = AGENT_ACTION_DICT[action_id]
        name = self.persons_dict[agent_id]["name"]

        action_duration = 0
        if self.state_dict[agent_id]["location_state"] == 1:
            return

        if  agent_action == "keep":
            return

        if agent_action in ["sleep", "relax", "eat", "walk", "read", "exercise", "work", "study", "buy"]:
            self.state_dict[agent_id]["action_time_start"] = self.timeHour
            self.state_dict[agent_id]["action"] = agent_action
            self.state_dict[agent_id]["action_id"] = action_id
            self.state_dict[agent_id]["action_duration"] = action_duration
            return
        if agent_action == "go_home":
            target_pos_name = "home"
            target_pos = self.state_dict[agent_id]["home_pos"]
            self.state_dict[agent_id]["target_location"] = 0
            self.state_dict[agent_id]["target_pos"] = target_pos
            success = self.client.moveAgent(agent_id, target_pos, 25, self.speed * random.uniform(1.2, 1.8))
        elif agent_action == "go_to_work":
            target_pos_name = random.choice(list(self.facility_dict["Gate"].keys()))
            target_pos = self.facility_dict["Gate"][target_pos_name]
            self.state_dict[agent_id]["target_location"] = 2
            self.state_dict[agent_id]["target_pos"] = target_pos
            success = self.client.moveAgent(agent_id, target_pos, 25, self.speed * random.uniform(1.2, 1.8))
        elif agent_action == "go_to_school":
            target_pos_name = random.choice(list(self.facility_dict["Gate"].keys()))
            target_pos = self.facility_dict["Gate"][target_pos_name]
            self.state_dict[agent_id]["target_location"] = 3
            self.state_dict[agent_id]["target_pos"] = target_pos
            success = self.client.moveAgent(agent_id, target_pos, 25, self.speed * random.uniform(1.2, 1.8))
        elif agent_action == "go_to_shop":
            target_pos_name = random.choice(list(self.facility_dict["Shop"].keys()))
            target_pos = self.facility_dict["Shop"][target_pos_name]
            self.state_dict[agent_id]["target_location"] = 4
            self.state_dict[agent_id]["target_pos"] = target_pos
            success = self.client.moveAgent(agent_id, target_pos, 25, self.speed * random.uniform(1.0, 1.5))
        elif agent_action == "go_to_community":
            target_pos_name = random.choice(list(self.facility_dict["ExercisePlace"].keys()))
            target_pos = self.facility_dict["ExercisePlace"][target_pos_name]
            self.state_dict[agent_id]["target_location"] = 5
            self.state_dict[agent_id]["target_pos"] = target_pos
            success = self.client.moveAgent(agent_id, target_pos, 25, self.speed * random.uniform(1.0, 1.5))

        
        self.state_dict[agent_id]["last_location_state"] = self.state_dict[agent_id]["location_state"]
        self.state_dict[agent_id]["location_state"] = 1
        self.state_dict[agent_id]["action_time_start"] = -1
        self.state_dict[agent_id]["action"] = agent_action
        self.state_dict[agent_id]["action_id"] = action_id
        self.state_dict[agent_id]["reach_time"] = -1
        
        return self.state_dict


    def set_scheduled_action(self,agent_id):

        self.timeHour = self.start_ts + ((self.action_interval * 60 * self.m_step) / 3600)

        self.day_of_week, self.hours, self.minutes, self.seconds = seconds_to_dhms(int(self.timeHour * 3600))

        hours = self.hours
        minutes = self.minutes
        cur_time = 60 * hours + minutes
        name = self.persons_dict[agent_id]["name"]
        action_time = self.persons_dict[agent_id]["action_time"]
        go_home_time = self.persons_dict[agent_id]["go_home_time"]
        probable_action = self.persons_dict[agent_id]["probable_action"]

        action_index = 0

        for i, key in enumerate(AGENT_ACTION_DICT_CN.keys()):
            if AGENT_ACTION_DICT_CN[key] == probable_action:
                action_index = i

        life_stage = self.persons_dict[agent_id]["life_stage"]

        # convert the action time in the form of minutes so we can compare it with the current time
        hours_part, separator, minutes_part = action_time.partition('时')
        minutes_str = minutes_part.split('分')[0]
        action_hour = int(hours_part)
        action_minutes = int(minutes_str)
        action_time_in_minutes = 60 * action_hour + action_minutes

        # convert the go home time in the form of minutes so we can compare it with the current time
        hours_part, separator, minutes_part = go_home_time.partition('时')
        minutes_str = minutes_part.split('分')[0]
        go_home_hour = int(hours_part)
        go_home_minutes = int(minutes_str)
        go_home_time_in_minutes = 60 * go_home_hour + go_home_minutes

        if "节假日" not in self.WEEK_HOLIDAY:
            # 按照30%概率执行
            random_number = random.randint(0, 100)
            if random_number < 30:
                if life_stage == "大学生" or life_stage == "未成年学生" or life_stage == "上班族":
                    # if the current time is later than the action_time and within 10 minutes (set it as the time_interval in run.py)
                    if cur_time > action_time_in_minutes and abs(cur_time - action_time_in_minutes) <= 10:
                        # set the scheduled action correctly as the probable_action
                        self.state_dict[agent_id]["scheduled_action"] = action_index
                        #print(f"The probable action is {probable_action}")

                    # if the current time is between the action time and go home time, and the location is not on_the_way
                    elif cur_time <= go_home_time_in_minutes and cur_time >= action_time_in_minutes + 5 and self.state_dict[agent_id]["location_state"] != 1:
                        # set working for adults and study for students

                        #如果学生在学校，则开始学习
                        #如果不在则变成去学校
                        if life_stage == "大学生" or life_stage == "未成年学生":
                            if self.state_dict[agent_id]["location_state"] == 3:
                                self.state_dict[agent_id]["scheduled_action"] = 8
                            else:
                                self.state_dict[agent_id]["scheduled_action"] = 12

                        # 如果上班族在公司，则开始工作
                        #如果不在则变成去工作地
                        elif life_stage == "上班族":
                            if self.state_dict[agent_id]["location_state"] == 2:
                                self.state_dict[agent_id]["scheduled_action"] = 7
                            else:
                                self.state_dict[agent_id]["scheduled_action"] = 11


                    else:
                        self.state_dict[agent_id]["scheduled_action"] = -1

                else:
                    self.state_dict[agent_id]["scheduled_action"] = -1
            else:
                self.state_dict[agent_id]["scheduled_action"] = -1
        else:
            self.state_dict[agent_id]["scheduled_action"] = -1

        return self.state_dict

    def get_action_masks(self, agent_id, idx):
        if "action_time_start" in self.state_dict[agent_id].keys() \
                and self.state_dict[agent_id]["action_time_start"] != -1 \
                and self.timeHour - self.state_dict[agent_id]["action_time_start"] < self.state_dict[agent_id]["action_duration"]:
            return np.array(LOCATION_ACTION_MASK[1])

        location_action_mask = np.array(LOCATION_ACTION_MASK[self.state_dict[agent_id]["location_state"]])
        life_stage_action_mask = np.array(LIFE_STAGE_ACTION_MASK[LIFE_STAGE_DICT[self.persons_dict[agent_id]["life_stage"]]])

        result = np.bitwise_and(location_action_mask, life_stage_action_mask)

        if self.state_dict[agent_id]["location_state"] != 1 \
                and "last_action_id" in self.state_dict[agent_id].keys() \
                and self.state_dict[agent_id]["last_action_id"] in (10, 11, 12, 13, 14):  # 上一帧动作是去往xx地方, 到达后则不能再预测保持动作
            result[1] = 0

        ## schedule rule
        self.state_dict[agent_id]["in_schedule"] = False
        life_stage = self.persons_dict[agent_id]["life_stage"]

        action_time = self.persons_dict[agent_id]["action_time"]
        go_home_time = self.persons_dict[agent_id]["go_home_time"]
        probable_action = self.persons_dict[agent_id]["probable_action"]

        # convert the action time in the form of minutes so we can compare it with the current time
        hours_part, separator, minutes_part = action_time.partition('时')
        minutes_str = minutes_part.split('分')[0]
        action_hour = int(hours_part)
        action_minutes = int(minutes_str)
        action_time_in_minutes = 60 * action_hour + action_minutes

        # convert the go home time in the form of minutes so we can compare it with the current time
        hours_part, separator, minutes_part = go_home_time.partition('时')
        minutes_str = minutes_part.split('分')[0]
        go_home_hour = int(hours_part)
        go_home_minutes = int(minutes_str)
        go_home_time_in_minutes = 60 * go_home_hour + go_home_minutes

        #self.day_of_week, self.hours, self.minutes, self.seconds = seconds_to_dhms(int(self.timeHour * 3600))

        hours = self.hours
        minutes = self.minutes
        #print(f"The current time is {hours} : {minutes}")
        cur_time = 60 * hours + minutes

        if "节假日" not in self.WEEK_HOLIDAY:
            if cur_time > action_time_in_minutes and cur_time < go_home_time_in_minutes:
                self.state_dict[agent_id]["in_schedule"] = True

                if life_stage in ("未成年学生", "大学生"):
                    if self.state_dict[agent_id]["location_state"] == 3:
                        result = np.array(SCHEDULE_ACTION_MASK["study"])
                    else:
                        result = np.array(SCHEDULE_ACTION_MASK["go_to_school"])

                if life_stage in ("上班族"):
                    if self.state_dict[agent_id]["location_state"] == 2:
                        result = np.array(SCHEDULE_ACTION_MASK["work"])
                    else:
                        result = np.array(SCHEDULE_ACTION_MASK["go_to_work"])

                if life_stage in ("退休"):
                    if self.state_dict[agent_id]["location_state"] == 5:
                        result = np.array(SCHEDULE_ACTION_MASK["walk_or_exercise"])
                    else:
                        result = np.array(SCHEDULE_ACTION_MASK["go_to_community"])

                if life_stage in ("自由职业"):
                    if self.state_dict[agent_id]["location_state"] == 4:
                        result = np.array(SCHEDULE_ACTION_MASK["buy"])
                    else:
                        result = np.array(SCHEDULE_ACTION_MASK["go_to_shop"])

            if cur_time < 6.5 * 60 + random.randint(-30, 30) or cur_time > 23 * 60 + random.randint(-60, 60):
                if life_stage in ("未成年学生", "大学生", "上班族"):
                    if self.state_dict[agent_id]["location_state"] == 0:
                        result = np.array(SCHEDULE_ACTION_MASK["sleep"])
                    else:
                        result = np.array(SCHEDULE_ACTION_MASK["go_home"])


                
                result[4] = 0
                result[6] = 0
                result[9] = 0
                result[11] = 0
                result[12] = 0
                result[13] = 0
                result[14] = 0
        else:
            if cur_time < 7.0 * 60 + random.randint(-30, 30):
                if life_stage in ("未成年学生", "大学生", "上班族"):
                    if self.state_dict[agent_id]["location_state"] == 0:
                        result = np.array(SCHEDULE_ACTION_MASK["sleep"])
                    else:
                        result = np.array(SCHEDULE_ACTION_MASK["go_home"])

                result[4] = 0
                result[6] = 0
                result[9] = 0
                result[13] = 0
                result[14] = 0

            result[11] = 0
            result[12] = 0

        return result

    def get_duration_masks(self, agent_id):
        return np.ones(self.action_duration_size)


    def show_building_statistic_info(self):
        ## show build state
        statistic_info = {}
        for agent_id in self.agent_list:
            pos = self.state_dict[agent_id]["pos"]

            agent_point = Point(pos.position.x_val, pos.position.y_val)

            building_id = "_".join(agent_id.split("_")[1:4]) + "_wd"
            building_poly = Polygon([[x["x_val"], x["y_val"]] for x in self.building_polygon[building_id]])

            if building_poly.contains(agent_point):
                statistic_info.setdefault(building_id, [])
                statistic_info[building_id].append(self.state_dict[agent_id]["action"])

        for k in self.building_set:
            v = statistic_info.get(k, None)
            if v is None:
                self.client.setBuildingStats(k, k.split("YSHY_")[1], 0, "")
                continue
            total_num = len(v)
            self.client.setBuildingStats(k, k.split("YSHY_")[1], total_num, str(dict(Counter(v))))


    def close(self):
        print("start remove agents")
        self.client.removeAgents()
        #time.sleep(5)



if __name__ == "__main__":


    # deploy model
    onnx_model_path = "model.onnx"
    session = onnxruntime.InferenceSession(onnx_model_path)
    print(session.get_inputs()[0])
    print(session.get_inputs()[1])
    print(session.get_inputs()[2])
    #print(session.get_inputs()[3].name)
    print(session.get_outputs()[0].name)
    print(session.get_outputs()[1].name)


    SPEED = 10
    START_TIME = 0
    TOTAL_HOURS = 24
    TOTAL_AGENT_NUM = 512
    ACTION_DURATION_SIZE = 0
    ACTION_INTERVAL = 10
    IS_TRAINING = False
    
    TRAIN_TASK_NAME = "deploy_0930a"

    env = SimEnv(SPEED, START_TIME, TOTAL_HOURS, TOTAL_AGENT_NUM, ACTION_DURATION_SIZE, ACTION_INTERVAL, TRAIN_TASK_NAME, IS_TRAINING)
    obs, infos = env.reset()

    for i in range(int(TOTAL_HOURS * 60 / ACTION_INTERVAL)):
        action_dict = {}
        #states = [[0] * 6 for _ in range(TOTAL_AGENT_NUM)]
        agent_list = list(obs.keys())

        for idx, agent_id in enumerate(agent_list):
            agent_obs = obs[agent_id]["observations"]
            action_masks = obs[agent_id]["action_mask"]

            #print("{} obs: {} action_masks: {}".format(agent_id, agent_obs, action_masks))

            input_obs = np.array([agent_obs], dtype=np.float32)
            input_masks = np.array([action_masks], dtype=np.float32)

            inputs = {session.get_inputs()[0].name: input_masks, 
                      session.get_inputs()[1].name: input_obs,
                      session.get_inputs()[2].name: np.zeros((1), dtype=np.float32),
                      #session.get_inputs()[3].name: np.zeros((4, 6), dtype=np.float32),
                      }
            output, state = session.run(None, inputs)

            # def softmax(x):
            #     exp_x = np.exp(x)
            #     return exp_x / np.sum(exp_x)

            # output_softmax = softmax(output[0])
            #action = np.random.choice(len(output_softmax), p=output_softmax)

            
            action = np.argmax(output[0])
            action_dict[agent_id] = action
            #print("{} obs: {} action_masks: {} action: {}".format(agent_id, agent_obs, action_masks, action))

            #print("input_obs: {}".format(input_obs[0]))

        obs, rewards, terminateds, truncateds, infos = env.step(action_dict)
        if terminateds["__all__"]:
            break



