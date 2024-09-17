import gym
from gym import spaces
import numpy as np
import pygame
import sys
import time
from gym.utils import seeding
import task_config
import utils
from typing import Optional


class TrackingEnv(gym.Env):
    """
    要放进venv/lib/python3.8/site-packages/gym/envs/user中，并且配置_init_.py文件
    """
    Metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 20
    }

    def __init__(self, target_mode, obstacle_mode, test_flag=False, cl_flag=False, training_stage=1):
        """
        初始化函数，定义关键变量。

        参数:
        - target_mode: 目标运动模态，取值为 {'Fix','Ram','Nav'} 中的一个。
        - obstacle_mode: 障碍分布类型，取值为 {'None','Static','Dynamic'} 中的一个。
        - test_flag: 测试模式标志，True 表示进入测试模式，False 表示进入普通训练模式，默认为 False。
        - cl_flag: 课程学习模式标志，True 表示进入课程学习训练模式，默认为 False。
        - training_stage: 仅在课程学习模式下有效，指定训练阶段，取值为 {1,2,3,4,5} 中的一个，默认为 1。

        在初始化过程中，会设置环境的各种参数，包括地图尺寸、动作空间、观测空间等，并根据模式设置目标运动模态和障碍物分布类型。
        """
        # 初始化画布、角度、距离等变量
        self.canvas = None
        self.last_tracker_angle = None
        self.last_target_angle_error = None
        self.last_target_angle = None
        self.last_target_distance_error = None
        self.angle = None
        self.tracker = None
        self.target = None
        self.target_move_direction = None
        self.last_target_distance = None
        self.move_num = 0
        self.nav_point = None
        self.free_space = None
        self.blocking_space = None
        self.mask_flag = task_config.mask_flag
        # 设置地图尺寸和像素大小
        self.width = task_config.width
        self.height = task_config.height
        self.pixel_size = task_config.pixel_size
        self.moving_size = task_config.moving_size
        # 定义动作空间和观测空间
        self.action_space = spaces.Discrete(11)  # 从行驶方向-45度开始沿正方向旋转
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 13), dtype=np.float32) if self.mask_flag else spaces.Box(low=-1, high=1, shape=(1, 18), dtype=np.float32)
        # 初始化静态和动态障碍物列表
        self.static_obstacles = []
        self.dynamic_obstacles = []
        # 设置检测距离和角度的参数
        self.max_detection_distance = task_config.max_detection_distance
        self.best_distance = task_config.best_distance
        self.max_detection_angle = task_config.max_detection_angle
        self.best_angle = task_config.best_angle
        # 设置测试模式和课程学习模式的标志
        self.test_flag = test_flag  # 测试模式：指定目标和障碍，有限步长；普通训练模式：指定目标和障碍，无限步长
        self.cl_flag = cl_flag  # 课程学习模式：已有目标和障碍设置，无固定步长
        self.training_stage = training_stage  # 课程学习模式下定义

        # 根据模式设置目标运动模态和障碍物分布类型
        if self.cl_flag:
            # 课程学习训练模式下，根据自定义课程设置
            if self.training_stage == 1:
                self.target_mode = "Fix"
                self.obstacle_mode = "None"
            elif self.training_stage == 2:
                self.target_mode = "Fix"
                self.obstacle_mode = "Static"
            elif self.training_stage == 3:
                self.target_mode = "Ram"
                self.obstacle_mode = "Static"
            elif self.training_stage == 4:
                self.target_mode = "Ram"
                self.obstacle_mode = "Dynamic"
            elif self.training_stage == 5:
                self.target_mode = "Nav"
                self.obstacle_mode = "Dynamic"
        else:
            # 测试模式和普通训练模式下，目标运动模态和障碍分布类型由用户指定
            self.target_mode = target_mode
            self.obstacle_mode = obstacle_mode
        # 设置总步数
        self.total_steps = task_config.total_steps
        # 初始化窗口和时钟（用于渲染和计时）
        self.window = None
        self.clock = None
        # 初始化智能体和目标的轨迹
        self.tracker_trajectory = []
        self.target_trajectory = []
        # 设置随机种子
        self.seed()
        # 初始化step方法相关参数
        self.step_count = 0
        self.continuous_tracking_step = 0
        self.max_continuous_tracking_step = 0
        self.loss_step = 0
        self.success_flag = False
        self.collision_flag = False
        self.loss_flag = False
        self.collide_with_obstacle = False
        self.collide_with_target = False
        self.last_seen_target_distance = 1
        self.last_seen_target_angle = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        主交互函数，用于处理智能体的动作并更新环境状态。

        参数:
        - action: 智能体执行的动作编号。

        返回:
        - next_observation: 下一时刻的观测。
        - reward: 当前动作对应的奖励。
        - terminated: 是否因为达到终止条件而结束回合（正常结束）。
        - truncated: 是否提前结束回合（碰撞或丢失）。
        - info: 其他辅助信息，如步数、成功标志、碰撞标志等。
        """
        # 更新全局状态
        self.step_count += 1
        # 移动tracker
        new_pos = {'x': 0, 'y': 0}  # 初始化智能体的下一个位置
        old_pos = self.tracker  # 获取智能体当前位置
        d, alpha = utils.move(action)  # 根据动作获取移动距离和角度变化
        beta = (self.last_tracker_angle + alpha) % 360  # 更新智能体的全局朝向角度
        new_pos['x'] = old_pos['x'] + d * np.cos(np.deg2rad(beta))  # 根据角度和距离计算新位置
        new_pos['y'] = old_pos['y'] + d * np.sin(np.deg2rad(beta))
        # 确保智能体不超出地图边界
        if new_pos['x'] < 0:
            new_pos['x'] = 0
        if new_pos['x'] >= self.width:
            new_pos['x'] = self.width - self.pixel_size
        if new_pos['y'] < 0:
            new_pos['y'] = 0
        if new_pos['y'] >= self.height:
            new_pos['y'] = self.height - self.pixel_size
        self.tracker = new_pos  # 更新tracker位置
        self.angle = beta  # 更新tracker朝向

        # 计算智能体移动后与目标的相对位置和角度等辅助变量
        target_relative_distance, target_distance_error, target_relative_angle, target_angle_error, tracker_angle = (
            utils.get_relative_elements(self.target, self.tracker, self.angle))
        # 更新全局变量
        self.last_target_distance = target_relative_distance
        self.last_target_angle = target_relative_angle
        self.last_target_distance_error = target_distance_error
        self.last_target_angle_error = target_angle_error
        self.last_tracker_angle = tracker_angle

        # 测试模式限制算法运行长度
        if self.test_flag and self.step_count >= self.total_steps:
            self.success_flag = True

        # 计算奖励
        (reward, terminated, truncated, self.success_flag, self.collision_flag, self.loss_flag,
         self.continuous_tracking_step, self.loss_step, self.max_continuous_tracking_step) \
            = utils.reward_calculate(self.test_flag, self.last_target_distance, self.last_target_angle,
                                     self.continuous_tracking_step, self.loss_step, self.max_continuous_tracking_step,
                                     self.tracker, self.target, self.static_obstacles, self.dynamic_obstacles)

        # 根据目标模式移动目标
        if self.target_mode == "Nav":
            self.target, self.move_num, self.nav_point = utils.target_nav(self.tracker, self.target, self.move_num,
                                                                          self.nav_point, self.static_obstacles,
                                                                          self.dynamic_obstacles, self.step_count)
        elif self.target_mode == "Fix":
            self.target = utils.move_clockwise(self.target, self.target_move_direction, self.moving_size)
        elif self.target_mode == "Ram":
            # 向可行域随机移动
            for _ in range(100):
                possible_moves = np.random.randint(0, 8)
                old_pos = {'x': self.target['x'], 'y': self.target['y']}
                new_pos = utils.move_in_grid(old_pos, possible_moves, self.moving_size)
                if utils.is_free_space(self.tracker, self.static_obstacles, self.dynamic_obstacles, new_pos['x'],
                                       new_pos['y'], is_static=False, is_initialize=False):
                    self.target['x'] = new_pos['x']
                    self.target['y'] = new_pos['y']
                    break

        # 移动动态障碍物
        if self.obstacle_mode == "Dynamic":
            self.dynamic_obstacles = utils.dynamic_obstacle_move(self.dynamic_obstacles, self.moving_size, self.tracker,
                                                                 self.static_obstacles)  # 随机动态障碍移动

        # 记录智能体和目标的轨迹
        self.tracker_trajectory.append(
            (self.tracker['x'] + self.pixel_size // 2, self.tracker['y'] + self.pixel_size // 2))
        self.target_trajectory.append(
            (self.target['x'] + self.pixel_size // 2, self.target['y'] + self.pixel_size // 2))

        # 为下一次观测作准备
        target_relative_distance, target_distance_error, target_relative_angle, target_angle_error, tracker_angle \
            = utils.get_relative_elements(self.target, self.tracker, self.angle)
        # 更新历史记录
        self.last_target_distance = target_relative_distance
        self.last_target_angle = target_relative_angle

        # 更新渲染画面
        self.canvas = utils.get_canvas(self.target_mode, self.obstacle_mode, self.free_space, self.blocking_space,
                                       self.target, self.tracker, self.static_obstacles, self.dynamic_obstacles,
                                       self.tracker_trajectory, self.target_trajectory, self.nav_point,
                                       self.last_tracker_angle)
        # 更新状态信息
        next_observation, self.last_seen_target_distance, self.last_seen_target_angle = utils.get_observation(
            self.last_target_distance, self.last_target_angle, self.last_seen_target_distance,
            self.last_seen_target_angle, self.tracker, self.angle, self.static_obstacles, self.dynamic_obstacles)

        # 需要的回合辅助信息
        info = {
            "total_step": self.step_count, "max_continuous_tracking_step": self.max_continuous_tracking_step,
            "success_flag": self.success_flag, "loss_flag": self.loss_flag, "collision_flag": self.collision_flag,
            "distance_error": self.last_target_distance_error, "angle_error": self.last_target_angle_error,
            "pos_tracker": self.tracker, "pos_target": self.target, "tracker_angle": self.angle
        }

        # 返回局部观察空间和奖励等信息
        return next_observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, ):
        """
        重置环境到初始状态，并返回初始观测和辅助信息。

        参数:
        - seed: 随机种子，用于确保结果的可重复性。
        - options: 可选参数字典，用于提供额外的重置选项。

        返回:
        - next_observation: 初始观测。
        - info: 包含环境状态信息的字典。
        """
        # 地图设置
        self.free_space = utils.map_process()
        self.blocking_space = []
        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.nav_point = []
        self.target = {}
        self.tracker = {}
        # 障碍分布类型设置
        if self.target_mode == "Fix":
            # 设置屏蔽区域保证运行通路
            for i in range(9, 40):
                self.blocking_space.append({'x': 90, 'y': 10 * i})
                self.blocking_space.append({'x': 390, 'y': 10 * i})
                self.blocking_space.append({'x': 10 * i, 'y': 90})
                self.blocking_space.append({'x': 10 * i, 'y': 390})

        if self.obstacle_mode == "Static":
            # 生成静态障碍（均匀），若想生成非均匀的则可以使用utils.Generate_Static_Obstacles_Randomly()
            self.static_obstacles = utils.generate_static_obstacles(self.tracker, self.dynamic_obstacles, self.blocking_space)
            self.dynamic_obstacles = []

        elif self.obstacle_mode == "Dynamic":
            # 生成静态障碍
            self.static_obstacles = utils.generate_static_obstacles(self.tracker, self.dynamic_obstacles, self.blocking_space)
            # 初始化随机动态障碍物(4个)
            for _ in range(4):
                while True:
                    obstacle = np.random.choice(self.free_space, replace=True)
                    if utils.is_free_space(self.tracker, self.static_obstacles, self.dynamic_obstacles, obstacle['x'],
                                           obstacle['y'], is_static=False, is_initialize=True):
                        self.dynamic_obstacles.append({'x': obstacle['x'], 'y': obstacle['y']})
                        break

        # 目标运动模态设置
        if self.target_mode == "Fix":
            # 生成初始位置和运动方向
            self.target = np.random.choice([{'x': 90, 'y': 90}, {'x': 90, 'y': 390}, {'x': 390, 'y': 90}, {'x': 390, 'y': 390}])
            self.target_move_direction = np.random.choice(['clockwise', 'counterclockwise'])

        else:
            # 生成初始位置
            self.target = np.random.choice([{'x': 80, 'y': 60}, {'x': 380, 'y': 130}, {'x': 120, 'y': 380}, {'x': 390, 'y': 430}, {'x': 280, 'y': 280}, {'x': 200, 'y': 200}])
            if self.target_mode == "Nav":
                # 清零运动计数器，重新生成导航点
                self.move_num = 0
                self.nav_point, self.move_num = utils.generate_nav_point(self.tracker, self.target,
                                                                         self.static_obstacles, self.dynamic_obstacles,
                                                                         self.step_count)

        # 跟踪器位置初始化
        while True:
            self.tracker = {
                'x': self.target['x'] + np.random.randint(-3, 3) * self.pixel_size,
                'y': self.target['y'] + np.random.randint(-3, 3) * self.pixel_size
            }
            if utils.is_free_space(self.tracker, self.static_obstacles, self.dynamic_obstacles, self.tracker['x'],
                                   self.tracker['y'], is_static=False,
                                   is_initialize=True) and self.tracker != self.target:
                break

        # 记录初始位置
        self.tracker_trajectory.append(
            (self.tracker['x'] + self.pixel_size // 2, self.tracker['y'] + self.pixel_size // 2))
        self.target_trajectory.append(
            (self.target['x'] + self.pixel_size // 2, self.target['y'] + self.pixel_size // 2))

        # 初值计算
        # 计算目标与智能体的相对距离和误差
        dx = self.target['x'] - self.tracker['x']
        dy = self.target['y'] - self.tracker['y']
        target_relative_distance = np.sqrt(dx ** 2 + dy ** 2)
        target_distance_error = (target_relative_distance - self.best_distance) / self.max_detection_distance
        # 计算目标与智能体的相对角度和误差
        target_angle = np.degrees(np.arctan2(dy, dx)) % 360
        tracker_angle = target_angle  # 初始角度指向target
        self.angle = tracker_angle
        target_relative_angle = target_angle - tracker_angle  # 0
        if target_relative_angle > 180:
            target_relative_angle -= 360
        elif target_relative_angle < -180:
            target_relative_angle += 360
        target_angle_error = (target_relative_angle - self.best_angle) / self.max_detection_angle  # 0

        # 存储上一步内容
        self.last_target_distance = target_relative_distance
        self.last_target_distance_error = target_distance_error
        self.last_target_angle = target_relative_angle
        self.last_target_angle_error = target_angle_error
        self.last_tracker_angle = tracker_angle
        self.last_seen_target_distance = self.last_target_distance
        self.last_seen_target_angle = self.last_target_angle
        # 更新渲染画面
        self.canvas = utils.get_canvas(self.target_mode, self.obstacle_mode, self.free_space, self.blocking_space,
                                       self.target, self.tracker, self.static_obstacles, self.dynamic_obstacles,
                                       self.tracker_trajectory, self.target_trajectory, self.nav_point,
                                       self.last_tracker_angle)
        # 获取observation
        next_observation, self.last_seen_target_distance, self.last_seen_target_angle = utils.get_observation(
            self.last_target_distance, self.last_target_angle, self.last_seen_target_distance,
            self.last_seen_target_angle, self.tracker, self.angle, self.static_obstacles, self.dynamic_obstacles)

        # 需要的回合辅助信息
        info = {
            "total_step": self.step_count, "max_continuous_tracking_step": self.max_continuous_tracking_step,
            "success_flag": self.success_flag, "loss_flag": self.loss_flag, "collision_flag": self.collision_flag,
            "distance_error": self.last_target_distance_error, "angle_error": self.last_target_angle_error,
            "pos_tracker": self.tracker, "pos_target": self.target, "tracker_angle": self.angle
        }

        return next_observation, info

    def render(self, mode='human'):
        """
        渲染仿真环境的当前状态。

        参数:
        - mode: 渲染模式，'human' 表示在窗口中显示，其他值表示返回RGB数组。

        无返回值
        """
        # 如果是首次渲染并且模式为'human'，则初始化pygame窗口和时钟
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))  # 初始化一张画布
            pygame.display.set_caption("室外环境下动态目标跟踪")

        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        # 获取当前环境状态的渲染画面
        self.canvas = utils.get_canvas(self.target_mode, self.obstacle_mode, self.free_space, self.blocking_space,
                                       self.target, self.tracker, self.static_obstacles, self.dynamic_obstacles,
                                       self.tracker_trajectory, self.target_trajectory, self.nav_point,
                                       self.last_tracker_angle)
        # 根据渲染模式进行显示或存储
        if mode == "human":
            # 将绘制的canvas内容复制到窗口中
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()  # 处理事件队列
            pygame.display.update()  # 更新显示

            # 控制渲染的帧率
            self.clock.tick(self.Metadata["render_fps"])
        else:  # 如果模式不是'human'，则返回RGB数组
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def close(self):
        """
        关闭仿真环境并释放相关资源。

        无返回值
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    '''主函数'''
    env = gym.make('TrackingEnv-v0', target_mode="Ram", obstacle_mode="Static", test_flag=True)
    # 设置随机数种子
    # env.seed(1)
    # 设置总时间步数
    total_steps = 300

    # 开始测试
    for episode in range(1):  # 设置测试回合数
        # 重置环境
        observation = env.reset()
        env.render()
        terminated, truncated = False, False

        # 初始化累计奖励
        total_reward = 0

        for step in range(total_steps):
            # 在动作空间中选择一个动作
            print(step)
            action = env.action_space.sample()  # 这里仅作为示例，使用随机动作
            # action = 10
            env.render()
            time.sleep(0.1)
            # 与环境交互
            observation, reward, terminated, truncated, info = env.step(action)
            # print(observation,reward,done,info)
            # print(action)
            # print("episode：", episode + 1, "step:", step + 1, "info:", info)
            # 累计奖励
            total_reward += reward

            # 判断是否达到终止条件
            if terminated or truncated:
                env.render()
                time.sleep(1)
                print(
                    f"Episode {episode + 1} finished after {step + 1} steps with total reward {total_reward}because of {info}")
                break

    env.close()
    sys.exit()
