import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# 自定义无人机环境
class DronePathEnv(gym.Env):
    def __init__(self):
        super(DronePathEnv, self).__init__()

        # 环境参数
        self.max_speed = 2.0  # 最大速度
        self.dt = 0.1  # 时间步长
        self.goal_radius = 0.5  # 目标区域半径
        self.obs_radius = 0.3  # 障碍物半径
        self.max_steps = 200  # 最大步数

        # 状态空间：[x, y, vx, vy, goal_x, goal_y, obs1_x, obs1_y, obs2_x, obs2_y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))

        # 动作空间：[水平推力，垂直推力]（连续动作）
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        # 初始化状态
        self.reset()

        # 可视化设置
        self.fig = None
        self.ax = None

    def reset(self):
        # 初始位置
        self.pos = np.array([0.0, 0.0])

        # 随机目标位置（在4x4区域内）
        self.goal = np.random.uniform(-2, 2, size=2)

        # 随机障碍物位置
        self.obstacles = [
            np.random.uniform(-1.5, 1.5, size=2),
            np.random.uniform(-1.5, 1.5, size=2)
        ]

        # 初始速度
        self.vel = np.zeros(2)

        # 步数计数器
        self.steps = 0

        return self._get_obs()

    def _get_obs(self):
        # 构造状态向量
        return np.concatenate([
            self.pos,
            self.vel,
            self.goal,
            self.obstacles[0],
            self.obstacles[1]
        ])

    def step(self, action):
        # 解析动作（限制在[-1,1]范围内）
        thrust = np.clip(action, -1, 1)

        # 更新速度（考虑推力系数0.5）
        self.vel += thrust * 0.5 * self.dt
        self.vel = np.clip(self.vel, -self.max_speed, self.max_speed)

        # 更新位置
        self.pos += self.vel * self.dt

        # 计算奖励
        reward = 0.0
        done = False

        # 到目标距离
        dist_to_goal = np.linalg.norm(self.pos - self.goal)

        # 到达目标奖励
        if dist_to_goal < self.goal_radius:
            reward += 10.0
            done = True

        # 接近目标奖励
        reward += (1.0 - dist_to_goal / 4.0) * 0.1

        # 碰撞检测
        for obs in self.obstacles:
            if np.linalg.norm(self.pos - obs) < self.obs_radius:
                reward -= 5.0
                done = True

        # 时间惩罚
        reward -= 0.01

        # 步数限制
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)

        # 绘制无人机
        self.ax.add_patch(Circle(self.pos, 0.2, color='blue'))

        # 绘制目标
        self.ax.add_patch(Circle(self.goal, self.goal_radius, color='green', alpha=0.3))

        # 绘制障碍物
        for obs in self.obstacles:
            self.ax.add_patch(Circle(obs, self.obs_radius, color='red'))

        plt.draw()
        plt.pause(0.01)

    def close(self):
        """新增方法：关闭图形资源"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# DDPG Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 输出在[-1,1]范围
        )

    def forward(self, state):
        return self.net(state)


# DDPG Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


# DDPG算法实现
class DDPG:
    def __init__(self, state_dim, action_dim):
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 软更新系数
        self.batch_size = 64  # 批量大小
        self.buffer_size = 100000  # 经验回放大小

        # 网络初始化
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 经验回放
        self.buffer = deque(maxlen=self.buffer_size)

        # 噪声生成器
        self.noise_std = 0.1

    def get_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # 从缓冲中采样
        batch = random.sample(self.buffer, self.batch_size)
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.FloatTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch]).unsqueeze(1)

        # Critic更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor更新
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 软更新目标网络
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


# 训练函数
def train_ddpg(env, agent, episodes=1000):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

        # 定期测试并渲染
        if episode % 50 == 0:
            test_agent(env, agent, render=True)

        print(f"Episode: {episode}, Reward: {episode_reward:.2f}")

    return rewards


# 测试函数
def test_agent(env, agent, render=False):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, add_noise=False)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if render:
            env.render()
# 新增：测试结束后关闭渲染窗口
    if render:
        env.close()
        plt.close('all')  # 关闭所有Matplotlib图形


if __name__ == "__main__":
    # 初始化环境和智能体
    env = DronePathEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPG(state_dim, action_dim)

    # 开始训练
    rewards = train_ddpg(env, agent, episodes=1000)

    # 训练结束后关闭环境
    env.close()

    # 绘制奖励曲线
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()