from rl_env.path_env import RlGame
# import pygame
# from assignment import constants as C
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle as pkl
shoplistfile_test = 'D:/drlTest/UAV-path-planning-main\Multi-UAVs path planning\path planning\sample_compare'  #保存文件数据所在文件的文件名
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
N_Agent=3
M_Enemy=0
# RENDER=True
RENDER=False
TRAIN_NUM = 1
TEST_EPIOSDE=100
env = RlGame(n=N_Agent,m=M_Enemy,render=RENDER).unwrapped
state_number=7+(N_Agent-1)
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
EP_MAX = 500
EP_LEN = 1000

def main():
    run(env)
def run(env):
    print('随机测试中...')
    action = np.zeros((N_Agent+M_Enemy, action_number))
    win_times = 0
    average_FKR=0
    average_timestep=0
    average_integral_V = 0
    average_integral_V0 = 0
    average_integral_V1 = 0
    average_integral_V2 = 0
    average_integral_U = 0
    average_integral_U0 = 0
    average_integral_U1 = 0
    average_integral_U2 = 0
    all_ep_V,all_ep_U,all_ep_T,all_ep_F=[],[],[],[]
    for j in range(TEST_EPIOSDE):
        state = env.reset()
        total_rewards = 0
        integral_V = 0
        integral_V0 = 0
        integral_V1 = 0
        integral_V2 = 0
        integral_U = 0
        integral_U0 = 0
        integral_U1 = 0
        integral_U2 = 0
        v,v1=[],[]
        for timestep in range(EP_LEN):
            for i in range(N_Agent+M_Enemy):
                action[i] = env.action_space.sample()
            # action[0] = aa.choose_action(state[0])
            # action[1] = bb.choose_action(state[1])
            new_state, reward,done,win,team_counter= env.step(action)  # 执行动作
            if win[0] and win[1] and win[2]:
                win_times += 1
            v.append(state[0][2])
            v1.append(state[1][2])
            integral_V += state[0][2]
            integral_V0 += state[0][2]
            integral_V1 += state[1][2]
            integral_V2 += state[2][2]
            integral_U += abs(action[0]).sum()
            integral_U0 += action[0][0] ** 2 * action[0][1] ** 2
            integral_U1 += action[1][0] ** 2 * action[1][1] ** 2
            integral_U2 += action[2][0] ** 2 * action[2][1] ** 2
            total_rewards += reward.mean()
            state = new_state
            if RENDER:
                env.render()
            if done:
                break
        FKR=team_counter/timestep
        average_FKR += FKR
        average_timestep += timestep
        average_integral_V += integral_V
        average_integral_V0 += integral_V0
        average_integral_V1 += integral_V1
        average_integral_V2 += integral_V2
        average_integral_U += integral_U
        average_integral_U0 += integral_U0
        average_integral_U1 += integral_U1
        average_integral_U2 += integral_U2
        print("Score", total_rewards)
        all_ep_V.append(integral_V)
        all_ep_U.append(integral_U)
        all_ep_T.append(timestep)
        all_ep_F.append(FKR)
        # print('最大编队保持率',FKR)
        # print('最短飞行时间',timestep)
        # print('最短飞行路程', integral_V)
        # print('最小能量损耗', integral_U)
        # plt.plot(np.arange(len(v)), v)
        # plt.plot(np.arange(len(v1)), v1)
        # plt.show()
    print('任务完成率', win_times / TEST_EPIOSDE)
    # print('平均最大编队保持率', average_FKR / TEST_EPIOSDE)
    print('平均最短飞行时间', average_timestep / TEST_EPIOSDE)
    print('平均最短飞行路程', average_integral_V / TEST_EPIOSDE)
    print('hero0平均最短飞行路程', average_integral_V0 / TEST_EPIOSDE)
    print('hero1平均最短飞行路程', average_integral_V1 / TEST_EPIOSDE)
    print('hero2平均最短飞行路程', average_integral_V2 / TEST_EPIOSDE)
    print('平均最小能量损耗', average_integral_U / TEST_EPIOSDE)
    print('hero0平均最小能量损耗', average_integral_U0 / TEST_EPIOSDE)
    print('hero1平均最小能量损耗', average_integral_U1 / TEST_EPIOSDE)
    print('hero2平均最小能量损耗', average_integral_U2 / TEST_EPIOSDE)
    # d = {"all_ep_V": all_ep_V,"all_ep_U": all_ep_U,"all_ep_T": all_ep_T,"all_ep_F": all_ep_F,}
    # f = open(shoplistfile_test, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
    # pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
    # f.close()
    env.close()

if __name__ == '__main__':
    main()