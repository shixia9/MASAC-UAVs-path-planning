import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math


# ===================== 环境设置 =====================
class MultiUAVEnv:
    def __init__(self, num_uavs=3, render=False):
        self.num_uavs = num_uavs
        self.render_mode = render
        self.width = 1000  # 环境宽度
        self.height = 800  # 环境高度

        # 初始化参数
        self.max_speed = 30  # 最大速度 (像素/步)
        self.safe_distance = 40  # 安全距离
        self.goal_radius = 50  # 目标半径

        # 目标位置 (随机生成)
        self.goal_pos = np.array([random.randint(800, 900), random.randint(600, 700)])

        # 障碍物设置
        self.obstacles = [
            {'pos': [400, 300], 'radius': 60},
            {'pos': [600, 500], 'radius': 40}
        ]

        # 重置环境
        self.reset()

    def reset(self):
        # 随机初始化无人机位置 (左下角区域)
        self.uavs = []
        for _ in range(self.num_uavs):
            pos = np.array([random.uniform(50, 200),
                            random.uniform(50, 200)])
            angle = random.uniform(0, 2 * math.pi)
            self.uavs.append({
                'pos': pos,
                'vel': np.zeros(2),
                'theta': angle,
                'done': False,
                'reward': 0.0
            })
        return self._get_obs()

    def _get_obs(self):
        # 构建每个无人机的观测向量
        observations = []
        for i in range(self.num_uavs):
            uav = self.uavs[i]

            # 自身状态 [x, y, vx, vy]
            obs = [uav['pos'][0] / self.width,
                   uav['pos'][1] / self.height,
                   uav['vel'][0] / self.max_speed,
                   uav['vel'][1] / self.max_speed]

            # 目标相对位置
            goal_vec = (self.goal_pos - uav['pos']) / np.array([self.width, self.height])
            obs.extend(goal_vec.tolist())

            # 最近障碍物信息
            min_obstacle_dist = 1.0
            for obs in self.obstacles:
                dist = np.linalg.norm(uav['pos'] - obs['pos']) / (self.width + self.height)
                if dist < min_obstacle_dist:
                    min_obstacle_dist = dist
                    obstacle_dir = (obs['pos'] - uav['pos']) / np.array([self.width, self.height])
            obs.extend([min_obstacle_dist, obstacle_dir[0], obstacle_dir[1]])

            # 其他无人机相对位置 (最近的两个)
            other_uavs = []
            for j in range(self.num_uavs):
                if j != i:
                    rel_pos = (self.uavs[j]['pos'] - uav['pos']) / np.array([self.width, self.height])
                    other_uavs.append((rel_pos, np.linalg.norm(rel_pos)))
            other_uavs.sort(key=lambda x: x[1])
            for k in range(min(2, len(other_uavs))):
                obs.extend(other_uavs[k][0].tolist())
                obs.append(other_uavs[k][1])

            observations.append(np.array(obs))
        return np.array(observations)

    def step(self, actions):
        # 处理动作 (归一化的加速度向量)
        rewards = np.zeros(self.num_uavs)
        dones = [False] * self.num_uavs
        infos = {}

        for i in range(self.num_uavs):
            if self.uavs[i]['done']:
                continue

            # 更新速度 (限制最大速度)
            action = np.clip(actions[i], -1, 1) * 5  # 缩放加速度
            self.uavs[i]['vel'] = np.clip(
                self.uavs[i]['vel'] + action,
                -self.max_speed, self.max_speed
            )

            # 更新位置
            new_pos = self.uavs[i]['pos'] + self.uavs[i]['vel']
            new_pos = np.clip(new_pos, [0, 0], [self.width, self.height])
            self.uavs[i]['pos'] = new_pos

            # 计算奖励
            reward = 0.0

            # 目标奖励
            goal_dist = np.linalg.norm(self.goal_pos - new_pos)
            if goal_dist < self.goal_radius:
                reward += 100.0
                self.uavs[i]['done'] = True
                dones[i] = True
            else:
                reward += 1.0 / (goal_dist + 1e-5)  # 距离奖励

            # 障碍物惩罚
            for obs in self.obstacles:
                obs_dist = np.linalg.norm(new_pos - obs['pos'])
                if obs_dist < obs['radius']:
                    reward -= 50.0
                    self.uavs[i]['done'] = True
                    dones[i] = True
                elif obs_dist < obs['radius'] + 50:
                    reward -= 10.0 / (obs_dist - obs['radius'] + 1e-5)

            # 无人机间碰撞检测
            for j in range(self.num_uavs):
                if j != i and not self.uavs[j]['done']:
                    uav_dist = np.linalg.norm(new_pos - self.uavs[j]['pos'])
                    if uav_dist < 2 * self.safe_distance:
                        reward -= 5.0
                        if uav_dist < self.safe_distance:
                            reward -= 100.0
                            self.uavs[i]['done'] = True
                            dones[i] = True

            rewards[i] = reward

        return self._get_obs(), rewards, dones, infos


# ===================== MASAC 算法实现 =====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)


class MASAC:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents) for _ in range(num_agents)]

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=3e-4) for critic in self.critics]

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.batch_size = 256
        self.memory = deque(maxlen=1000000)

    def select_action(self, state, evaluate=False):
        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                s = torch.FloatTensor(state[i]).unsqueeze(0)
                dist = self.actors[i](s)
                if evaluate:
                    action = dist.mean
                else:
                    action = dist.sample()
                actions.append(action.numpy()[0])
        return np.array(actions)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # 采样批次数据
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # 更新Critic
        for i in range(self.num_agents):
            # 计算目标Q值
            with torch.no_grad():
                next_actions = []
                next_log_probs = []
                for j in range(self.num_agents):
                    dist = self.actors[j](next_states[:, j, :])
                    next_action = dist.rsample()
                    next_log_prob = dist.log_prob(next_action).sum(1, keepdim=True)
                    next_actions.append(next_action)
                    next_log_probs.append(next_log_prob)

                next_actions = torch.cat(next_actions, dim=1)
                next_log_probs = torch.cat(next_log_probs, dim=1)

                target_q1, target_q2 = self.target_critics[i](
                    next_states.view(-1, self.num_agents * state_dim),
                    next_actions.view(-1, self.num_agents * action_dim)
                )
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs.sum(1, keepdim=True)
                target_q = rewards[:, i].unsqueeze(1) + (1 - dones[:, i].unsqueeze(1)) * self.gamma * target_q

            # 当前Q值
            current_q1, current_q2 = self.critics[i](
                states.view(-1, self.num_agents * state_dim),
                actions.view(-1, self.num_agents * action_dim)
            )
            critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

            # 优化Critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # 更新Actor
        for i in range(self.num_agents):
            dist = self.actors[i](states[:, i, :])
            new_actions = dist.rsample()
            log_probs = dist.log_prob(new_actions).sum(1, keepdim=True)

            # 计算新动作的Q值
            new_action_list = []
            for j in range(self.num_agents):
                if j == i:
                    new_action_list.append(new_actions)
                else:
                    new_action_list.append(actions[:, j, :])

            new_actions_all = torch.cat(new_action_list, dim=1)
            q1, q2 = self.critics[i](
                states.view(-1, self.num_agents * state_dim),
                new_actions_all.view(-1, self.num_agents * action_dim)
            )
            q = torch.min(q1, q2)

            actor_loss = (-q + self.alpha * log_probs).mean()

            # 优化Actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # 软更新目标网络
        for i in range(self.num_agents):
            for param, target_param in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


# ===================== 训练流程 =====================
if __name__ == "__main__":
    env = MultiUAVEnv(num_uavs=3)
    state_dim = env._get_obs().shape[1]  # 获取状态维度
    action_dim = 2  # 二维加速度控制

    agent = MASAC(num_agents=3, state_dim=state_dim, action_dim=action_dim)

    episodes = 1000
    max_steps = 500

    for ep in range(episodes):
        state = env.reset()
        total_rewards = np.zeros(3)

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.save_experience(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_rewards += np.array(reward)

            if all(done):
                break

        print(f"Episode {ep + 1}, Total Reward: {total_rewards}, Steps: {step + 1}")

        # 定期保存模型
        if (ep + 1) % 100 == 0:
            torch.save({
                'actor': [actor.state_dict() for actor in agent.actors],
                'critic': [critic.state_dict() for critic in agent.critics]
            }, f"masac_model_ep{ep + 1}.pth")