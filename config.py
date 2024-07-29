import argparse
import torch
import pygame

# 算法相关变量
task = "TrackingEnv-v0"
reward_threshold = None
seed = 1
eps_test = 0.05
eps_train = 0.1
buffer_size = 5e5
lr = 1e-3
gamma = 0.9
num_atoms = 51
v_min = -10.0
v_max = 10.0
noisy_std = 0.1
n_step = 3
target_update_freq = 100
epoch = 500
step_per_epoch = 10000
step_per_collect = 200
update_per_step = 0.05
batch_size = 256
hidden_sizes = [128, 128]
training_num = 10
test_num = 10
logdir = "performance_test_new"
render = 0.1
prioritized_replay = False
alpha = 0.6
beta = 0.4
beta_final = 1.0
resume = False
device = "cuda" if torch.cuda.is_available() else "cpu"
save_interval = 4

# 环境变量
width = 500
height = 500
pixel_size = 10
moving_size = 5
mask_flag = False
collision_penalty = -100
loss_penalty = -50
max_detection_distance = 50
best_distance = 20
max_detection_angle = 45
best_angle = 0
max_loss_step = 50
total_steps = 500
min_obstacle_size = 4
max_obstacle_size = 6
tracker_img = pygame.image.load("/home/ln/projects/gym_code/robot.png")
target_img = pygame.image.load("/home/ln/projects/gym_code/running_kid.png")
obstacle_kind_probability=[0.4, 0.1, 0.1, 0.2, 0.2]


'''
def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="TrackingEnv_v3-v0")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=5e5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)  
    parser.add_argument("--num-atoms", type=int, default=51)  # 分布式强化学习
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--noisy-std", type=float, default=0.1)  # 噪声网络
    parser.add_argument("--n-step", type=int, default=3)  # 预估
    parser.add_argument("--target-update-freq", type=int, default=100)  # 目标网络更新频率
    parser.add_argument("--epoch", type=int, default=500)  # 训练
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=200)
    parser.add_argument("--update-per-step", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)  # NN
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)  # 测试episode数=需要构建的测试环境数
    parser.add_argument("--logdir", type=str, default="performance_test_new")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument("--prioritized-replay", action="store_false",
                        default=True)  # 优先经验回放；action="store_true"是触发机制，方便命令行运行，等价于default=False
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")  # 恢复训练过的模型
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save-interval", type=int, default=4)

    return parser.parse_known_args()[0]
'''

