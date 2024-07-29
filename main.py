import os
import pickle
import pprint
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import RainbowPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear
import config

def Train():
    env = gym.make(config.task, CL=False, target_mode="Fix", obstacle_mode="Dynamic")
    config.state_shape = env.observation_space.shape
    config.action_shape = env.action_space.n
    # 指定奖励上限
    if config.reward_threshold is None:
        default_reward_threshold = {"TrackingEnv-v0": 500000}
        config.reward_threshold = default_reward_threshold.get(config.task, env.spec.reward_threshold)

    # 构建训练环境和测试环境
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(config.task, CL=False, target_mode="Fix", obstacle_mode="Dynamic") for _ in
         range(config.training_num)])
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(config.task, CL=False, target_mode="Fix", obstacle_mode="Dynamic") for _ in
         range(config.test_num)])

    # 设置随机数种子
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_envs.seed(config.seed)
    test_envs.seed(config.seed)

    # 构建网络
    def noisy_linear(x, y):
        return NoisyLinear(x, y, config.noisy_std)

    net = Net(
        config.state_shape,
        config.action_shape,
        hidden_sizes=config.hidden_sizes,
        device=config.device,
        softmax=True,
        num_atoms=config.num_atoms,
        dueling_param=({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}),
    )
    # 优化器
    optim = torch.optim.Adam(net.parameters(), lr=config.lr)
    # 策略
    policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=config.gamma,
        action_space=env.action_space,
        num_atoms=config.num_atoms,
        v_min=config.v_min,
        v_max=config.v_max,
        estimation_step=config.n_step,
        target_update_freq=config.target_update_freq,
    ).to(config.device)

    # buffer
    if config.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            config.buffer_size,
            buffer_num=len(train_envs),
            alpha=config.alpha,
            beta=config.beta,
            weight_norm=True,
        )
    else:
        buf = VectorReplayBuffer(config.buffer_size, buffer_num=len(train_envs))

    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # 预采集
    train_collector.collect(n_step=config.batch_size * config.training_num)

    # 设置数据存储路径
    log_path = os.path.join(config.logdir)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=config.save_interval)

    # 存储最优记录
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # 终止函数
    def stop_fn(mean_rewards):
        return mean_rewards >= config.reward_threshold

    # 设置训练eps
    def train_fn(epoch, env_step):
        if env_step <= 10000:
            policy.set_eps(config.eps_train)
        elif env_step <= 50000:
            eps = config.eps_train - (env_step - 10000) / 40000 * (0.9 * config.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * config.eps_train)
        # beta annealing, just a demo
        if config.prioritized_replay:
            if env_step <= 10000:
                beta = config.beta
            elif env_step <= 50000:
                beta = config.beta - (env_step - 10000) / 40000 * (config.beta - config.beta_final)
            else:
                beta = config.beta_final
            buf.set_beta(beta)

    # 设置测试eps
    def test_fn(epoch, env_step):
        policy.set_eps(config.eps_test)

    # 存储checkpoint
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(train_collector.buffer, f)
        return ckpt_path

    # 从历史记录中恢复训练（如果有设置）
    if config.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            policy.load_state_dict(checkpoint["model"])
            policy.optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                train_collector.buffer = pickle.load(f)
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=config.epoch,
            step_per_epoch=config.step_per_epoch,
            step_per_collect=config.step_per_collect,
            episode_per_test=config.test_num,
            batch_size=config.batch_size,
            update_per_step=config.update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=config.resume,
            save_checkpoint_fn=save_checkpoint_fn,
            ).run()


# 从历史记录恢复训练
def Train_Resume(args):
    config.resume = True
    Train()

# 使用优先经验回放训练
def Train_Prioritized():
    config.prioritized_replay = True
    config.resume = True
    Train()


if __name__ == "__main__":
    Train()

