import torch

task = "TrackingEnv-v0"  # 环境名称
reward_threshold = None  # 奖励阈值，用于评估算法性能，None表示没有预设阈值
seed = 1  # 随机种子，用于确保实验可重复性
eps_test = 0.05  # 测试阶段的epsilon值，用于epsilon-greedy策略
eps_train = 0.1  # 训练阶段的epsilon值，用于epsilon-greedy策略
buffer_size = 5e5  # 经验回放缓冲区的大小
lr = 1e-3  # 学习率
gamma = 0.9  # 折扣因子，用于计算未来奖励的当前价值
num_atoms = 51  # 分布式价值函数中原子的数量
v_min = -10.0  # 价值函数的最小值
v_max = 10.0  # 价值函数的最大值
noisy_std = 0.1  # 噪声标准差，用于添加噪声以打破对称性
n_step = 3  # n步回报，用于计算回报时考虑的步数
target_update_freq = 100  # 目标网络更新频率
epoch = 500  # 训练周期总数
step_per_epoch = 10000  # 每个训练周期内的步数
step_per_collect = 200  # 每收集一次数据的步数
update_per_step = 0.05  # 每步更新的频率
batch_size = 256  # 批处理大小
hidden_sizes = [128, 128]  # 神经网络的隐含层大小
training_num = 10  # 训练环境数量
test_num = 10  # 测试环境数量
logdir = "performance_test_new"  # 日志目录，用于保存训练和测试结果
render = 0.1  # 渲染频率，用于可视化训练过程
prioritized_replay = False  # 是否使用优先级经验回放
alpha = 0.6  # 优先级经验回放中的alpha参数，用于计算重要性采样权重
beta = 0.4  # 优先级经验回放中的beta参数，用于初始的重要性采样
beta_final = 1.0  # 优先级经验回放中的beta参数的最终值
resume = False  # 是否从上次训练中断的地方继续
device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用的设备，优先使用GPU
save_interval = 4  # 保存模型的间隔
