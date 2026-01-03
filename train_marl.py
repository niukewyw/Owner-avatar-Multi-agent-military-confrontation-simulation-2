#!/usr/bin/env python3
"""
多智能体强化学习训练脚本

基于迭代自我对弈的训练流程：
1. 初始化阶段：使用规则AI作为起点
2. 交替训练敌我双方策略
3. 评估与监控训练效果
4. 支持MAPPO算法，后续可扩展QMIX/VDN

使用示例:
    python train_marl.py --map attack_focused --total_iterations 50 --episodes_per_iter 100
    python train_marl.py --map defense_focused --opponent_pool_size 10 --eval_episodes 50
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import json

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from environment import MilitaryEnvironment
from demo_astar import generate_a_star_actions  # 作为规则AI的替代选择


class MAPPOAgent(nn.Module):
    """MAPPO智能体网络"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(MAPPOAgent, self).__init__()

        # 观察编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 策略头（actor）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # 价值头（critic）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        """前向传播"""
        features = self.obs_encoder(obs)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, obs, deterministic=False):
        """根据观察选择动作"""
        with torch.no_grad():
            # obs 已经是一个张量，直接使用
            if not isinstance(obs, torch.Tensor):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            else:
                obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs
            action_probs, _ = self.forward(obs_tensor)

            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()

            return action

    def evaluate_actions(self, obs, actions):
        """评估动作的价值（用于训练）"""
        # obs 已经是一个张量，直接使用
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
        else:
            obs_tensor = obs
        # actions 已经是一个张量，直接使用
        if not isinstance(actions, torch.Tensor):
            actions_tensor = torch.LongTensor(actions).to(self.device)
        else:
            actions_tensor = actions

        action_probs, values = self.forward(obs_tensor)
        action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1))

        # 计算熵（用于正则化）
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()

        return action_log_probs, values.squeeze(), entropy


class MAPPOTrainer:
    """MAPPO训练器"""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建策略网络
        self.policy = MAPPOAgent(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        print(f"MAPPO训练器初始化完成 - 设备: {self.device}")

    def update_policy(self, rollouts):
        """更新策略网络"""
        # 从rollouts中提取数据
        obs_list = [r['obs'] for r in rollouts]
        obs = torch.stack(obs_list).to(self.device)
        actions = torch.LongTensor([r['action'] for r in rollouts]).to(self.device)
        old_log_probs = torch.FloatTensor([r['log_prob'] for r in rollouts]).to(self.device)
        returns = torch.FloatTensor([r['return'] for r in rollouts]).to(self.device)
        advantages = torch.FloatTensor([r['advantage'] for r in rollouts]).to(self.device)

        # 计算新策略的log概率和价值
        new_log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)

        # 计算策略损失 (PPO clipped objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 计算价值损失
        value_loss = ((values - returns) ** 2).mean()

        # 计算总损失
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }

    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计 (GAE)"""
        advantages = []
        gae = 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[step + 1]

            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        return advantages

    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }, path)
        print(f"模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"模型已加载自: {path}")
        else:
            print(f"模型文件不存在: {path}")


class MARLTrainer:
    """多智能体强化学习训练器"""

    def __init__(self, map_config: str, use_astar: bool = False):
        self.map_config = map_config
        self.use_astar = use_astar

        # 创建环境
        self.env = MilitaryEnvironment(map_config)

        # 获取智能体数量
        self.num_ally_agents = len(self.env.map.ally_agents)
        self.num_enemy_agents = len(self.env.map.enemy_agents)
        self.total_agents = self.num_ally_agents + self.num_enemy_agents

        # 定义观察空间和动作空间
        # 测试一个智能体的观察来确定维度
        test_obs = self.env.get_agent_observation(self.env.map.ally_agents[0])
        flattened_obs = self._flatten_observation(test_obs)
        self.obs_dim = len(flattened_obs)
        self.action_dim = 5  # 移动动作：不动、上、下、左、右

        # 创建训练器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ally_trainer = MAPPOTrainer(self.obs_dim, self.action_dim)
        self.enemy_trainer = MAPPOTrainer(self.obs_dim, self.action_dim)

        # 训练统计
        self.training_stats = {
            'iterations': 0,
            'ally_wins': 0,
            'enemy_wins': 0,
            'draws': 0,
            'total_episodes': 0
        }

        print(f"MARL训练器初始化完成")
        print(f"地图: {os.path.basename(map_config)}")
        print(f"智能体: {self.num_ally_agents} 我方 + {self.num_enemy_agents} 敌方")
        print(f"观察维度: {len(self._flatten_observation(test_obs))}, 动作维度: {self.action_dim}")

    def get_rule_actions(self, team: str) -> Dict[str, Dict]:
        """获取规则AI动作"""
        if self.use_astar:
            return generate_a_star_actions(team, self.env)
        else:
            # 使用基础规则AI
            return self.env._generate_rule_actions(team)

    def collect_rollouts(self, trainer: MAPPOTrainer, opponent_actions_func,
                        num_episodes: int = 10, max_steps: int = 100) -> List[Dict]:
        """收集训练数据"""
        rollouts = []

        for episode in range(num_episodes):
            # 重置环境
            obs = self.env.reset()
            done = False
            episode_steps = 0

            # 存储episode数据
            episode_obs = []
            episode_actions = []
            episode_log_probs = []
            episode_rewards = []
            episode_values = []
            episode_dones = []

            while not done and episode_steps < max_steps:
                # 为训练方智能体生成动作
                ally_actions = {}
                enemy_actions = opponent_actions_func()  # 调用函数生成对手动作

                # 收集当前观察
                current_obs = []
                current_actions = []
                current_log_probs = []
                current_values = []

                # 训练方使用策略网络
                for agent in (self.env.map.ally_agents if trainer == self.ally_trainer else self.env.map.enemy_agents):
                    if agent.alive:
                        agent_obs = self._get_agent_observation(agent.id)
                        action = trainer.policy.get_action(agent_obs)

                        # 记录观察和动作
                        current_obs.append(agent_obs)
                        current_actions.append(action)

                        # 获取log概率和价值
                        with torch.no_grad():
                            # agent_obs 已经是一个张量，直接使用
                            obs_tensor = agent_obs.unsqueeze(0) if agent_obs.dim() == 1 else agent_obs
                            action_probs, value = trainer.policy(obs_tensor)
                            log_prob = torch.log(action_probs[0, action])
                            current_log_probs.append(log_prob.item())
                            current_values.append(value.item())

                        # 生成环境动作
                        if trainer == self.ally_trainer:
                            ally_actions[agent.id] = {'move': self._action_to_move(action), 'attack': False, 'special': False}
                        else:
                            enemy_actions[agent.id] = {'move': self._action_to_move(action), 'attack': False, 'special': False}

                # 执行动作
                observations, rewards, done, info = self.env.step(ally_actions, enemy_actions)

                # 记录奖励和done状态
                team_reward = rewards['ally'] if trainer == self.ally_trainer else rewards['enemy']
                episode_rewards.append(team_reward)
                episode_dones.append(done)

                # 存储当前步骤的数据
                episode_obs.extend(current_obs)
                episode_actions.extend(current_actions)
                episode_log_probs.extend(current_log_probs)
                episode_values.extend(current_values)

                episode_steps += 1

            # 简化处理：为每个观测创建基本的rollout数据
            if episode_obs:
                # 为每个观测分配对应的奖励（简化：使用episode的总奖励）
                total_reward = sum(episode_rewards) if episode_rewards else 0.0

                for i in range(len(episode_obs)):
                    rollouts.append({
                        'obs': episode_obs[i],
                        'action': episode_actions[i],
                        'log_prob': episode_log_probs[i],
                        'value': episode_values[i] if i < len(episode_values) else 0.0,
                        'advantage': total_reward,  # 简化为总奖励
                        'return': total_reward
                    })

        return rollouts

    def _get_agent_observation(self, agent_id: str) -> torch.Tensor:
        """获取智能体的观察"""
        # 找到对应的智能体
        for agent in self.env.map.get_all_agents():
            if agent.id == agent_id and agent.alive:
                obs_dict = self.env.get_agent_observation(agent)
                flattened = self._flatten_observation(obs_dict)
                # 确保维度正确
                if len(flattened) != self.obs_dim:
                    print(f"警告: 观察维度不匹配 - 期望{self.obs_dim}, 实际{len(flattened)}")
                    # 截断或填充到正确维度
                    if len(flattened) > self.obs_dim:
                        flattened = flattened[:self.obs_dim]
                    else:
                        flattened = np.pad(flattened, (0, self.obs_dim - len(flattened)))
                # 转换为torch张量并移动到正确的设备
                return torch.FloatTensor(flattened).to(self.device)

        # 如果找不到智能体，返回零向量
        return torch.zeros(self.obs_dim, dtype=torch.float32).to(self.device)

    def _action_to_move(self, action: int) -> Tuple[int, int]:
        """将动作索引转换为移动方向"""
        # 动作映射：0=不动, 1=上, 2=下, 3=左, 4=右
        move_map = {
            0: (0, 0),   # 不动
            1: (0, -1),  # 上
            2: (0, 1),   # 下
            3: (-1, 0),  # 左
            4: (1, 0)    # 右
        }
        return move_map.get(action, (0, 0))

    def train_iteration(self, ally_first: bool = True, episodes_per_iter: int = 50):
        """执行一次完整的训练迭代（A->B或B->A）"""

        if ally_first:
            # 步骤A：训练敌方（固定我方）
            print("训练敌方策略...")
            def get_ally_actions(): return self.get_rule_actions('ally')
            rollouts = self.collect_rollouts(self.enemy_trainer, get_ally_actions, episodes_per_iter)

            if rollouts:
                loss_info = self.enemy_trainer.update_policy(rollouts)
                print(f"敌方训练完成 - 损失: {loss_info['total_loss']:.3f}")
            # 步骤B：训练我方（固定敌方）
            print("训练我方策略...")
            def get_enemy_actions(): return self.get_rule_actions('enemy')
            rollouts = self.collect_rollouts(self.ally_trainer, get_enemy_actions, episodes_per_iter)

            if rollouts:
                loss_info = self.ally_trainer.update_policy(rollouts)
                print(f"我方训练完成 - 损失: {loss_info['total_loss']:.3f}")
        else:
            # 相反顺序
            print("训练我方策略...")
            def get_enemy_actions(): return self.get_rule_actions('enemy')
            rollouts = self.collect_rollouts(self.ally_trainer, get_enemy_actions, episodes_per_iter)

            if rollouts:
                loss_info = self.ally_trainer.update_policy(rollouts)
                print(f"我方训练完成 - 损失: {loss_info['total_loss']:.3f}")
            print("训练敌方策略...")
            def get_ally_actions(): return self.get_rule_actions('ally')
            rollouts = self.collect_rollouts(self.enemy_trainer, get_ally_actions, episodes_per_iter)

            if rollouts:
                loss_info = self.enemy_trainer.update_policy(rollouts)
                print(f"敌方训练完成 - 损失: {loss_info['total_loss']:.3f}")
    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """评估当前策略"""
        print(f"开始评估 ({num_episodes} episodes)...")

        ally_wins = 0
        enemy_wins = 0
        draws = 0
        total_rewards_ally = 0
        total_rewards_enemy = 0
        total_length = 0

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_length = 0
            episode_rewards_ally = 0
            episode_rewards_enemy = 0

            while not done and episode_length < 200:  # 最大200步
                # 生成动作
                ally_actions = {}
                enemy_actions = {}

                # 我方使用训练好的策略
                for agent in self.env.map.ally_agents:
                    if agent.alive:
                        agent_obs = self._get_agent_observation(agent.id)
                        action = self.ally_trainer.policy.get_action(agent_obs, deterministic=True)
                        ally_actions[agent.id] = {
                            'move': self._action_to_move(action),
                            'attack': False,
                            'special': False
                        }

                # 敌方使用训练好的策略
                for agent in self.env.map.enemy_agents:
                    if agent.alive:
                        agent_obs = self._get_agent_observation(agent.id)
                        action = self.enemy_trainer.policy.get_action(agent_obs, deterministic=True)
                        enemy_actions[agent.id] = {
                            'move': self._action_to_move(action),
                            'attack': False,
                            'special': False
                        }

                # 执行动作
                observations, rewards, done, info = self.env.step(ally_actions, enemy_actions)

                episode_rewards_ally += rewards['ally']
                episode_rewards_enemy += rewards['enemy']
                episode_length += 1

            # 统计结果
            total_rewards_ally += episode_rewards_ally
            total_rewards_enemy += episode_rewards_enemy
            total_length += episode_length

            if self.env.game_over:
                if self.env.winner == 'ally':
                    ally_wins += 1
                elif self.env.winner == 'enemy':
                    enemy_wins += 1
                else:
                    draws += 1

        # 计算统计结果
        results = {
            'ally_win_rate': ally_wins / num_episodes,
            'enemy_win_rate': enemy_wins / num_episodes,
            'draw_rate': draws / num_episodes,
            'avg_ally_reward': total_rewards_ally / num_episodes,
            'avg_enemy_reward': total_rewards_enemy / num_episodes,
            'avg_episode_length': total_length / num_episodes
        }

        print(f"我方胜率: {results['ally_win_rate']:.1f}")
        print(f"敌方胜率: {results['enemy_win_rate']:.1f}")
        print(f"平局率: {results['draw_rate']:.1f}")
        print(f"平均我方奖励: {results['avg_ally_reward']:.1f}")
        print(f"平均敌方奖励: {results['avg_enemy_reward']:.1f}")
        return results

    def _compute_gae_and_returns(self, rewards, values, dones, gamma, gae_lambda):
        """计算GAE和回报"""
        advantages = []
        returns = []

        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0  # 最后一步的next_value为0
            else:
                next_value = values[step + 1]

            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return advantages, returns

    def _flatten_observation(self, obs_dict: Dict) -> np.ndarray:
        """将观察字典展平为向量 - 固定维度"""
        # 固定尺寸的局部地图 (21x21 = 441)
        local_map = obs_dict['local_map']
        # 确保地图是21x21，如果不是则填充或裁剪
        if local_map.shape[0] != 21 or local_map.shape[1] != 21:
            # 创建21x21的地图
            fixed_map = np.zeros((21, 21), dtype=local_map.dtype)
            h, w = local_map.shape
            # 复制现有数据
            fixed_map[:min(21, h), :min(21, w)] = local_map[:min(21, h), :min(21, w)]
            map_flat = fixed_map.flatten()
        else:
            map_flat = local_map.flatten()

        # 固定数量的可见敌人位置 (10个敌人 x 2维 = 20)
        enemy_pos = np.zeros(20, dtype=np.float32)
        for i, enemy in enumerate(obs_dict['visible_enemies'][:10]):
            enemy_pos[i*2:i*2+2] = enemy

        # 其他特征 (4个)
        other_features = np.array([
            obs_dict['health_ratio'],
            1.0 if obs_dict['can_attack'] else 0.0,
            obs_dict['position'][0] / 300.0,  # 归一化位置
            obs_dict['position'][1] / 300.0
        ], dtype=np.float32)

        # 合并所有特征: 441 + 20 + 4 = 465
        obs_vector = np.concatenate([map_flat, enemy_pos, other_features])
        return obs_vector

    def save_models(self, iteration: int, save_dir: str = "models"):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)

        ally_path = os.path.join(save_dir, f"ally_iter_{iteration}.pth")
        enemy_path = os.path.join(save_dir, f"enemy_iter_{iteration}.pth")

        self.ally_trainer.save_model(ally_path)
        self.enemy_trainer.save_model(enemy_path)

        print(f"模型已保存 - 迭代 {iteration}")

    def load_models(self, iteration: int, save_dir: str = "models"):
        """加载模型"""
        ally_path = os.path.join(save_dir, f"ally_iter_{iteration}.pth")
        enemy_path = os.path.join(save_dir, f"enemy_iter_{iteration}.pth")

        self.ally_trainer.load_model(ally_path)
        self.enemy_trainer.load_model(enemy_path)

        print(f"模型已加载 - 迭代 {iteration}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多智能体强化学习训练')

    parser.add_argument('--map', type=str, required=True,
                       choices=['attack_focused', 'defense_focused', 'balanced'],
                       help='选择地图配置')

    parser.add_argument('--total_iterations', type=int, default=20,
                       help='总训练迭代次数')

    parser.add_argument('--episodes_per_iter', type=int, default=50,
                       help='每轮迭代的episode数量')

    parser.add_argument('--eval_episodes', type=int, default=20,
                       help='评估时使用的episode数量')

    parser.add_argument('--use_astar', action='store_true',
                       help='使用A*算法作为规则AI')

    parser.add_argument('--save_dir', type=str, default='models',
                       help='模型保存目录')

    parser.add_argument('--load_iteration', type=int, default=None,
                       help='从指定迭代加载模型继续训练')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    print("=== 多智能体强化学习训练 ===")
    print(f"地图: {args.map}")
    print(f"总迭代数: {args.total_iterations}")
    print(f"每轮episodes: {args.episodes_per_iter}")
    print(f"评估episodes: {args.eval_episodes}")
    print(f"使用A*: {args.use_astar}")
    print("-" * 50)

    # 检查地图配置文件
    map_config_path = f"maps/{args.map}.json"
    if not os.path.exists(map_config_path):
        print(f"错误: 地图配置文件 '{map_config_path}' 不存在!")
        return 1

    try:
        # 创建训练器
        trainer = MARLTrainer(map_config_path, use_astar=args.use_astar)

        # 如果指定了加载迭代，从检查点继续训练
        if args.load_iteration is not None:
            trainer.load_models(args.load_iteration, args.save_dir)

        # 训练循环
        start_time = time.time()

        for iteration in range(1, args.total_iterations + 1):
            print(f"\n=== 迭代 {iteration}/{args.total_iterations} ===")

            # 训练一轮（交替训练敌我）
            ally_first = (iteration % 2 == 1)  # 奇数轮我方先训练，偶数轮敌方先训练
            trainer.train_iteration(ally_first, args.episodes_per_iter)

            # 每5轮评估一次
            if iteration % 5 == 0:
                eval_results = trainer.evaluate(args.eval_episodes)

                # 保存模型
                trainer.save_models(iteration, args.save_dir)

                # 检查是否达到均衡状态
                ally_win_rate = eval_results['ally_win_rate']
                if 0.45 <= ally_win_rate <= 0.55:
                    print(f"检测到均衡状态 - 我方胜率: {ally_win_rate:.1f}")
        # 最终评估
        print("\n=== 最终评估 ===")
        final_results = trainer.evaluate(args.eval_episodes)
        trainer.save_models(args.total_iterations, args.save_dir)

        # 输出训练总结
        end_time = time.time()
        duration = end_time - start_time
        print("\n=== 训练完成 ===")
        print(f"训练时间: {duration:.2f}秒")
        print(f"总迭代数: {args.total_iterations}")
        print(f"总episodes: {args.total_iterations * args.episodes_per_iter * 2}")

        return 0

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)