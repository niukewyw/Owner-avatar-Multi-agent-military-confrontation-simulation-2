"""
多智能体强化学习训练框架
基于 MAPPO 算法实现迭代自我对弈
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from collections import deque
import matplotlib.pyplot as plt

from environment import MilitaryEnvironment


class ActorCritic(nn.Module):
    """Actor-Critic网络"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Actor网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic网络（价值）
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        """前向传播"""
        action_probs = self.actor(obs)
        value = self.critic(obs)
        return action_probs, value

    def get_action(self, obs):
        """采样动作"""
        action_probs, value = self.forward(obs)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


class MAPPOAgent:
    """MAPPO智能体"""

    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99,
                 clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 训练数据存储
        self.reset_memory()

    def reset_memory(self):
        """重置记忆"""
        self.obs_memory = []
        self.action_memory = []
        self.log_prob_memory = []
        self.reward_memory = []
        self.done_memory = []
        self.value_memory = []

    def store_transition(self, obs: np.ndarray, action: int, log_prob: float,
                        reward: float, done: bool, value: float):
        """存储转移"""
        self.obs_memory.append(obs)
        self.action_memory.append(action)
        self.log_prob_memory.append(log_prob)
        self.reward_memory.append(reward)
        self.done_memory.append(done)
        self.value_memory.append(value)

    def compute_returns(self, rewards: List[float], dones: List[bool], last_value: float) -> List[float]:
        """计算GAE回报"""
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = self.value_memory[step + 1]

            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - self.value_memory[step]
            gae = delta + self.gamma * 0.95 * gae  # lambda = 0.95
            returns.insert(0, gae + self.value_memory[step])

        return returns

    def update(self, last_obs: Optional[np.ndarray] = None):
        """更新策略"""
        if len(self.obs_memory) == 0:
            return

        # 计算回报
        if last_obs is not None:
            _, _, last_value = self.get_action(last_obs)
        else:
            last_value = 0

        returns = self.compute_returns(self.reward_memory, self.done_memory, last_value)

        # 转换为tensor
        obs_tensor = torch.FloatTensor(np.array(self.obs_memory)).to(self.device)
        action_tensor = torch.LongTensor(self.action_memory).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_prob_memory).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages = returns_tensor - torch.FloatTensor(self.value_memory).to(self.device)

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(10):  # 10个epoch
            # 前向传播
            action_probs, values = self.model(obs_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()

            # 计算比率
            ratios = torch.exp(new_log_probs - old_log_probs)

            # 计算损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        self.reset_memory()

    def get_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """获取动作（用于推理）"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(obs_tensor)
        return action, log_prob, value

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class MARLTrainer:
    """多智能体强化学习训练器"""

    def __init__(self, map_config_path: str, obs_dim: int = 100, action_dim: int = 5,
                 num_agents: int = 18, max_episode_steps: int = 200):
        self.map_config_path = map_config_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.max_episode_steps = max_episode_steps

        # 创建环境
        self.env = MilitaryEnvironment(map_config_path)

        # 创建智能体
        self.ally_agents = [MAPPOAgent(obs_dim, action_dim) for _ in range(num_agents)]
        self.enemy_agents = [MAPPOAgent(obs_dim, action_dim) for _ in range(num_agents)]

        # 训练统计
        self.stats = {
            'episode': 0,
            'ally_wins': 0,
            'enemy_wins': 0,
            'draws': 0,
            'ally_rewards': [],
            'enemy_rewards': [],
            'episode_lengths': []
        }

    def collect_rollout(self, ally_agents: List[MAPPOAgent], enemy_agents: List[MAPPOAgent],
                       num_episodes: int = 10) -> Dict[str, Any]:
        """收集rollout数据"""
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []

        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_rewards = {'ally': 0, 'enemy': 0}
            episode_length = 0

            for step in range(self.max_episode_steps):
                # 我方动作
                ally_actions = {}
                ally_obs = []
                ally_action_list = []
                ally_log_probs = []
                ally_values = []

                for i, agent in enumerate(self.env.map.ally_agents):
                    if agent.alive:
                        # 简化的观察（实际应该从agent.get_observation获取）
                        agent_obs = np.random.randn(self.obs_dim)  # 临时随机观察
                        action, log_prob, value = ally_agents[i].get_action(agent_obs)

                        ally_actions[agent.id] = {'move': (action % 3 - 1, action // 3 - 1),
                                                'attack': action % 2 == 1,
                                                'special': False}

                        ally_obs.append(agent_obs)
                        ally_action_list.append(action)
                        ally_log_probs.append(log_prob)
                        ally_values.append(value)

                # 敌方动作
                enemy_actions = {}
                enemy_obs = []
                enemy_action_list = []
                enemy_log_probs = []
                enemy_values = []

                for i, agent in enumerate(self.env.map.enemy_agents):
                    if agent.alive:
                        # 简化的观察
                        agent_obs = np.random.randn(self.obs_dim)  # 临时随机观察
                        action, log_prob, value = enemy_agents[i].get_action(agent_obs)

                        enemy_actions[agent.id] = {'move': (action % 3 - 1, action // 3 - 1),
                                                  'attack': action % 2 == 1,
                                                  'special': False}

                        enemy_obs.append(agent_obs)
                        enemy_action_list.append(action)
                        enemy_log_probs.append(log_prob)
                        enemy_values.append(value)

                # 执行动作
                observations, rewards, done, info = self.env.step(ally_actions, enemy_actions)

                # 存储经验
                if ally_obs:
                    all_obs.extend(ally_obs)
                    all_actions.extend(ally_action_list)
                    all_log_probs.extend(ally_log_probs)
                    all_values.extend(ally_values)
                    all_rewards.extend([rewards['ally']] * len(ally_obs))
                    all_dones.extend([done] * len(ally_obs))

                episode_rewards['ally'] += rewards['ally']
                episode_rewards['enemy'] += rewards['enemy']
                episode_length += 1

                if done:
                    break

            # 记录episode统计
            if episode_rewards['ally'] > episode_rewards['enemy']:
                self.stats['ally_wins'] += 1
            elif episode_rewards['enemy'] > episode_rewards['ally']:
                self.stats['enemy_wins'] += 1
            else:
                self.stats['draws'] += 1

            self.stats['ally_rewards'].append(episode_rewards['ally'])
            self.stats['enemy_rewards'].append(episode_rewards['enemy'])
            self.stats['episode_lengths'].append(episode_length)

        return {
            'obs': np.array(all_obs),
            'actions': np.array(all_actions),
            'log_probs': np.array(all_log_probs),
            'rewards': np.array(all_rewards),
            'dones': np.array(all_dones),
            'values': np.array(all_values)
        }

    def train_ally_agents(self, opponent_agents: List[MAPPOAgent], num_updates: int = 100):
        """训练我方智能体"""
        print("训练我方智能体...")

        for update in range(num_updates):
            # 收集数据
            rollout_data = self.collect_rollout(self.ally_agents, opponent_agents, num_episodes=5)

            # 更新每个智能体
            for i, agent in enumerate(self.ally_agents):
                # 这里简化处理，实际应该根据智能体ID匹配数据
                if len(rollout_data['obs']) > 0:
                    # 简化的更新逻辑
                    agent.store_transition(
                        rollout_data['obs'][0],  # 简化：只用第一个观察
                        rollout_data['actions'][0],
                        rollout_data['log_probs'][0],
                        rollout_data['rewards'][0],
                        rollout_data['dones'][0],
                        rollout_data['values'][0]
                    )
                    agent.update()

            if (update + 1) % 20 == 0:
                print(f"  更新 {update + 1}/{num_updates} 完成")

    def train_enemy_agents(self, opponent_agents: List[MAPPOAgent], num_updates: int = 100):
        """训练敌方智能体"""
        print("训练敌方智能体...")

        for update in range(num_updates):
            # 收集数据
            rollout_data = self.collect_rollout(opponent_agents, self.enemy_agents, num_episodes=5)

            # 更新每个智能体
            for i, agent in enumerate(self.enemy_agents):
                if len(rollout_data['obs']) > 0:
                    agent.store_transition(
                        rollout_data['obs'][0],
                        rollout_data['actions'][0],
                        rollout_data['log_probs'][0],
                        rollout_data['rewards'][0],
                        rollout_data['dones'][0],
                        rollout_data['values'][0]
                    )
                    agent.update()

            if (update + 1) % 20 == 0:
                print(f"  更新 {update + 1}/{num_updates} 完成")

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """评估当前策略"""
        print(f"评估策略 ({num_episodes} episodes)...")

        eval_stats = {
            'ally_wins': 0,
            'enemy_wins': 0,
            'draws': 0,
            'ally_avg_reward': 0,
            'enemy_avg_reward': 0,
            'avg_length': 0
        }

        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_rewards = {'ally': 0, 'enemy': 0}
            episode_length = 0

            for step in range(self.max_episode_steps):
                # 简化的随机动作（实际应该使用训练好的策略）
                ally_actions = {}
                for agent in self.env.map.ally_agents:
                    if agent.alive:
                        ally_actions[agent.id] = {'move': (np.random.randint(-1, 2), np.random.randint(-1, 2)),
                                                'attack': np.random.random() < 0.3,
                                                'special': False}

                enemy_actions = {}
                for agent in self.env.map.enemy_agents:
                    if agent.alive:
                        enemy_actions[agent.id] = {'move': (np.random.randint(-1, 2), np.random.randint(-1, 2)),
                                                  'attack': np.random.random() < 0.3,
                                                  'special': False}

                observations, rewards, done, info = self.env.step(ally_actions, enemy_actions)

                episode_rewards['ally'] += rewards['ally']
                episode_rewards['enemy'] += rewards['enemy']
                episode_length += 1

                if done:
                    break

            # 统计结果
            if episode_rewards['ally'] > episode_rewards['enemy']:
                eval_stats['ally_wins'] += 1
            elif episode_rewards['enemy'] > episode_rewards['ally']:
                eval_stats['enemy_wins'] += 1
            else:
                eval_stats['draws'] += 1

            eval_stats['ally_avg_reward'] += episode_rewards['ally']
            eval_stats['enemy_avg_reward'] += episode_rewards['enemy']
            eval_stats['avg_length'] += episode_length

        # 计算平均值
        eval_stats['ally_avg_reward'] /= num_episodes
        eval_stats['enemy_avg_reward'] /= num_episodes
        eval_stats['avg_length'] /= num_episodes

        print(f"评估结果:")
        print(".1f"        print(".1f"        print(".1f"        print(".1f"
        return eval_stats

    def iterative_self_play(self, num_iterations: int = 20, updates_per_iteration: int = 50,
                           eval_episodes: int = 20):
        """迭代自我对弈训练"""
        print("开始迭代自我对弈训练...")

        for iteration in range(num_iterations):
            print(f"\n=== 迭代 {iteration + 1}/{num_iterations} ===")

            # 步骤A：训练敌方策略
            print("步骤A: 训练敌方策略...")
            self.train_enemy_agents(self.ally_agents, updates_per_iteration)

            # 步骤B：训练我方策略
            print("步骤B: 训练我方策略...")
            self.train_ally_agents(self.enemy_agents, updates_per_iteration)

            # 评估
            eval_stats = self.evaluate(eval_episodes)

            # 保存检查点
            self.save_checkpoint(f"checkpoints/iteration_{iteration + 1}")

            print(f"迭代 {iteration + 1} 完成")

        print("训练完成！")

    def save_checkpoint(self, path: str):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'iteration': self.stats['episode'],
            'ally_agents': [agent.model.state_dict() for agent in self.ally_agents],
            'enemy_agents': [agent.model.state_dict() for agent in self.enemy_agents],
            'stats': self.stats
        }

        torch.save(checkpoint, f"{path}.pt")
        print(f"检查点已保存: {path}.pt")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(f"{path}.pt")
        self.stats = checkpoint['stats']

        for i, state_dict in enumerate(checkpoint['ally_agents']):
            self.ally_agents[i].model.load_state_dict(state_dict)

        for i, state_dict in enumerate(checkpoint['enemy_agents']):
            self.enemy_agents[i].model.load_state_dict(state_dict)

        print(f"检查点已加载: {path}.pt")


if __name__ == "__main__":
    # 创建训练器
    trainer = MARLTrainer("maps/attack_focused.json")

    # 开始训练
    trainer.iterative_self_play(num_iterations=5, updates_per_iteration=10, eval_episodes=5)
