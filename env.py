import gym
from gym import spaces
import numpy as np
import pygame
import sys
import time
from gym.utils import seeding
import config
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

    def __init__(self, target_mode, obstacle_mode, TEST=False, CL=False, training_stage=1):
        """
        初始化函数，定义关键变量

        :param string target_mode: 目标运动模态，in {'Fix','Ram','Nav'}
        :param string obstacle_mode: 障碍分布类型，in {'None','Static','Dynamic'}
        :param bool TEST: 测试模式，True:进入测试模式，False：进入普通训练模式，default to False
        :param bool CL: 课程学习模式，True：进入课程学习训练模式,default to False
        :param int training_stage: 仅在课程学习模式起效，指定训练阶段，in{1,2,3,4,5}

        """
        self.Mask_Flag = config.mask_flag

        self.Width = config.width
        self.Height = config.height
        self.Pixel_Size = config.pixel_size
        self.Moving_Size = config.moving_size
        self.action_space = spaces.Discrete(11)  # 从行驶方向-45度开始沿正方向旋转
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 13),
                                            dtype=np.float32) if self.Mask_Flag else spaces.Box(low=-1, high=1,
                                                                                                shape=(1, 18),
                                                                                                dtype=np.float32)

        self.Static_Obstacles = []
        self.Dynamic_Obstacles = []

        self.Collision_Penalty = config.collision_penalty
        self.Loss_Penalty = config.loss_penalty
        self.Max_Detection_Distance = config.max_detection_distance
        self.Best_Distance = config.best_distance
        self.Max_Detection_Angle = config.max_detection_angle
        self.Best_Angle = config.best_angle
        self.Max_Loss_Step = config.max_loss_step
        self.TEST = TEST  # 测试模式：指定目标和障碍，有限步长；普通训练模式：指定目标和障碍，无限步长
        self.CL = CL  # 课程学习模式：已有目标和障碍设置，无固定步长
        self.Training_Stage = training_stage  # 课程学习模式下定义

        # 根据模式设置目标运动模态和障碍物分布类型
        if self.CL:
            # 课程学习训练模式下，根据自定义课程设置
            if self.Training_Stage == 1:
                self.Target_Mode = "Fix"
                self.Obstacle_Mode = "None"
            elif self.Training_Stage == 2:
                self.Target_Mode = "Fix"
                self.Obstacle_Mode = "Static"
            elif self.Training_Stage == 3:
                self.Target_Mode = "Ram"
                self.Obstacle_Mode = "Static"
            elif self.Training_Stage == 4:
                self.Target_Mode = "Ram"
                self.Obstacle_Mode = "Dynamic"
            elif self.Training_Stage == 5:
                self.Target_Mode = "Nav"
                self.Obstacle_Mode = "Dynamic"
        else:
            # 测试模式和普通训练模式下，目标运动模态和障碍分布类型由用户指定
            self.Target_Mode = target_mode
            self.Obstacle_Mode = obstacle_mode

        self.Total_Steps = config.total_steps

        self.Window = None
        self.Clock = None

        # 轨迹
        self.Tracker_Trajectory = []
        self.Target_Trajectory = []
        # 静态障碍物大小
        self.Min_Obstacle_Size = config.min_obstacle_size  # min_obstacle_size
        self.Max_Obstacle_Size = config.max_obstacle_size  # max_obstacle_size

        self.seed()

        # step方法相关参数
        self.Step_Count = 0
        self.Continuous_Tracking_Step = 0
        self.Max_Continuous_Tracking_Step = 0
        self.Loss_Step = 0
        self.Success_Flag = False
        self.CollisionFlag = False
        self.Loss_Flag = False
        self.Collide_With_Obstacle = False
        self.Collide_With_Target = False
        self.Last_Seen_Target_Distance = 1
        self.Last_Seen_Target_Angle = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        主交互函数
        :param action: 智能体动作
        :return: next_observation下一时刻观察/状态, reward当前动作对应奖励, terminated, truncated回合终止标志, info其他辅助信息
        """
        # 更新全局状态
        self.Step_Count += 1
        # 移动tracker
        new_pos = {'x': 0, 'y': 0}  # tracker下一时刻位置
        old_pos = self.Tracker  # tracker当前时刻位置
        d, alpha = utils.Move(action)  # 获取当前动作对应的局部移动距离和角度
        beta = (self.Last_Tracker_Angle + alpha) % 360  # 计算全局角度
        new_pos['x'] = old_pos['x'] + d * np.cos(np.deg2rad(beta))  # 计算下一时刻位置
        new_pos['y'] = old_pos['y'] + d * np.sin(np.deg2rad(beta))
        # 移动不能超出地图边界
        if new_pos['x'] < 0:
            new_pos['x'] = 0
        if new_pos['x'] >= self.Width:
            new_pos['x'] = self.Width - self.Pixel_Size
        if new_pos['y'] < 0:
            new_pos['y'] = 0
        if new_pos['y'] >= self.Height:
            new_pos['y'] = self.Height - self.Pixel_Size
        self.Tracker = new_pos  # 更新tracker位置
        self.Angle = beta  # 更新tracker朝向

        # tracker移动之后计算辅助变量
        target_relative_distance, target_distance_error, target_relative_angle, target_angle_error, tracker_angle = (
            utils.Get_Relative_Elements(self.Target, self.Tracker, self.Angle))
        # 更新全局变量
        self.Last_Target_Distance = target_relative_distance
        self.Last_Target_Angle = target_relative_angle
        self.Last_Target_Distance_Error = target_distance_error
        self.Last_Target_Angle_Error = target_angle_error
        self.Last_Tracker_Angle = tracker_angle

        # 测试模式限制算法运行长度
        if self.TEST == True and self.Step_Count >= self.Total_Steps:
            truncated = True
            self.Success_Flag = True

        # tracker移动之后计算奖励
        (reward, terminated, truncated, self.Success_Flag, self.CollisionFlag, self.Loss_Flag,
         self.Continuous_Tracking_Step, self.Loss_Step, self.Max_Continuous_Tracking_Step) \
            = utils.Reward_Calculate(self.TEST, self.Last_Target_Distance, self.Last_Target_Angle,
                                     self.Continuous_Tracking_Step, self.Loss_Step, self.Max_Continuous_Tracking_Step,
                                     self.Tracker, self.Target, self.Static_Obstacles, self.Dynamic_Obstacles)

        # 移动目标
        if self.Target_Mode == "Nav":
            self.Target, self.Move_Num, self.Nav_Point = utils.Target_Nav(self.Tracker, self.Target, self.Move_Num,
                                                                          self.Nav_Point, self.Static_Obstacles,
                                                                          self.Dynamic_Obstacles, self.Step_Count)
        elif self.Target_Mode == "Fix":
            self.Target = utils.Move_Clockwise(self.Target, self.Target_Move_Direction, self.Moving_Size)
        elif self.Target_Mode == "Ram":
            # 向可行域随机移动
            for _ in range(100):
                possible_moves = np.random.randint(0, 8)
                old_pos = {'x': self.Target['x'], 'y': self.Target['y']}
                new_pos = utils.Move_In_Grid(old_pos, possible_moves, self.Moving_Size)
                if utils.Is_Free_Space(self.Tracker, self.Static_Obstacles, self.Dynamic_Obstacles, new_pos['x'],
                                       new_pos['y'], is_static=False, is_initialize=False):
                    self.Target['x'] = new_pos['x']
                    self.Target['y'] = new_pos['y']
                    break

        # 移动动态障碍物
        if self.Obstacle_Mode == "Dynamic":
            # 移动动态障碍物
            self.Dynamic_Obstacles = utils.Dynamic_Obstacle_Move(self.Dynamic_Obstacles, self.Moving_Size, self.Tracker,
                                                                 self.Static_Obstacles)  # 随机动态障碍移动

        # 记录需要的轨迹
        self.Tracker_Trajectory.append(
            (self.Tracker['x'] + self.Pixel_Size // 2, self.Tracker['y'] + self.Pixel_Size // 2))
        self.Target_Trajectory.append(
            (self.Target['x'] + self.Pixel_Size // 2, self.Target['y'] + self.Pixel_Size // 2))

        # 为下一次观测作准备
        target_relative_distance, target_distance_error, target_relative_angle, target_angle_error, tracker_angle \
            = utils.Get_Relative_Elements(self.Target, self.Tracker, self.Angle)
        # 更新历史记录
        self.Last_Target_Distance = target_relative_distance
        self.Last_Target_Angle = target_relative_angle

        # 更新渲染画面
        self.Canvas = utils.Get_Canvas(self.Target_Mode, self.Obstacle_Mode, self.Free_space, self.Blocking_Space,
                                       self.Target, self.Tracker, self.Static_Obstacles, self.Dynamic_Obstacles,
                                       self.Tracker_Trajectory, self.Target_Trajectory, self.Nav_Point,
                                       self.Last_Tracker_Angle)
        # 更新状态信息
        next_observation, self.Last_Seen_Target_Distance, self.Last_Seen_Target_Angle = utils.Get_Observation(
            self.Last_Target_Distance, self.Last_Target_Angle, self.Last_Seen_Target_Distance,
            self.Last_Seen_Target_Angle, self.Tracker, self.Angle, self.Static_Obstacles, self.Dynamic_Obstacles)

        # 需要的回合辅助信息
        info = {
            "total_step": self.Step_Count, "max_continuous_tracking_step": self.Max_Continuous_Tracking_Step,
            "success_flag": self.Success_Flag, "loss_flag": self.Loss_Flag, "collision_flag": self.CollisionFlag,
            "distance_error": self.Last_Target_Distance_Error, "angle_error": self.Last_Target_Angle_Error,
            "pos_tracker": self.Tracker, "pos_target": self.Target, "tracker_angle": self.Angle
        }

        # 返回局部观察空间和奖励等信息
        return next_observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, ):
        """
        环境初始化
        :return: (next_observation,info)
        """
        # 地图设置
        self.Free_space = utils.Map_Process()
        self.Blocking_Space = []
        self.Static_Obstacles = []
        self.Dynamic_Obstacles = []
        self.Nav_Point = []
        self.Target = []
        self.Tracker = []
        # 障碍分布类型设置
        if self.Target_Mode == "Fix":
            # 设置屏蔽区域保证运行通路
            for i in range(9, 40):
                self.Blocking_Space.append({'x': 90, 'y': 10 * i})
                self.Blocking_Space.append({'x': 390, 'y': 10 * i})
                self.Blocking_Space.append({'x': 10 * i, 'y': 90})
                self.Blocking_Space.append({'x': 10 * i, 'y': 390})

        if self.Obstacle_Mode == "Static":
            # 生成静态障碍（均匀），若想生成非均匀的则可以使用utils.Generate_Static_Obstacles_Randomly()
            self.Static_Obstacles = utils.Generate_Static_Obstacles(self.Tracker, self.Static_Obstacles,
                                                                    self.Dynamic_Obstacles, self.Blocking_Space)
            self.Dynamic_Obstacles = []

        elif self.Obstacle_Mode == "Dynamic":
            # 生成静态障碍
            self.Static_Obstacles = utils.Generate_Static_Obstacles(self.Tracker, self.Static_Obstacles,
                                                                    self.Dynamic_Obstacles, self.Blocking_Space)
            # 初始化随机动态障碍物(4个)
            for _ in range(4):
                while True:
                    obstacle = np.random.choice(self.Free_space, replace=True)
                    if utils.Is_Free_Space(self.Tracker, self.Static_Obstacles, self.Dynamic_Obstacles, obstacle['x'],
                                           obstacle['y'], is_static=False, is_initialize=True):
                        self.Dynamic_Obstacles.append({'x': obstacle['x'], 'y': obstacle['y']})
                        break

        # 目标运动模态设置
        if self.Target_Mode == "Fix":
            # 生成初始位置和运动方向
            self.Target = np.random.choice(
                [{'x': 90, 'y': 90}, {'x': 90, 'y': 390}, {'x': 390, 'y': 90}, {'x': 390, 'y': 390}])
            self.Target_Move_Direction = np.random.choice(['clockwise', 'counterclockwise'])

        else:
            # 生成初始位置
            self.Target = np.random.choice(
                [{'x': 80, 'y': 60}, {'x': 380, 'y': 130}, {'x': 120, 'y': 380}, {'x': 390, 'y': 430},
                 {'x': 280, 'y': 280}, {'x': 200, 'y': 200}])
            if self.Target_Mode == "Nav":
                # 清零运动计数器，重新生成导航点
                self.Move_Num = 0
                self.Nav_Point, self.Move_Num = utils.Generate_Nav_Point(self.Tracker, self.Target,
                                                                         self.Static_Obstacles, self.Dynamic_Obstacles,
                                                                         self.Step_Count)

        # 跟踪器位置初始化
        while True:
            self.Tracker = {
                'x': self.Target['x'] + np.random.randint(-3, 3) * self.Pixel_Size,
                'y': self.Target['y'] + np.random.randint(-3, 3) * self.Pixel_Size
            }
            if utils.Is_Free_Space(self.Tracker, self.Static_Obstacles, self.Dynamic_Obstacles, self.Tracker['x'],
                                   self.Tracker['y'], is_static=False,
                                   is_initialize=True) and self.Tracker != self.Target:
                break

        # 记录初始位置
        self.Tracker_Trajectory.append(
            (self.Tracker['x'] + self.Pixel_Size // 2, self.Tracker['y'] + self.Pixel_Size // 2))
        self.Target_Trajectory.append(
            (self.Target['x'] + self.Pixel_Size // 2, self.Target['y'] + self.Pixel_Size // 2))

        # 初值计算
        # 计算目标与智能体的相对距离和误差
        dx = self.Target['x'] - self.Tracker['x']
        dy = self.Target['y'] - self.Tracker['y']
        target_relative_distance = np.sqrt(dx ** 2 + dy ** 2)
        target_distance_error = (target_relative_distance - self.Best_Distance) / self.Max_Detection_Distance
        # 计算目标与智能体的相对角度和误差
        target_angle = np.degrees(np.arctan2(dy, dx)) % 360
        tracker_angle = target_angle  # 初始角度指向target
        self.Angle = tracker_angle
        target_relative_angle = target_angle - tracker_angle  # 0
        if target_relative_angle > 180:
            target_relative_angle -= 360
        elif target_relative_angle < -180:
            target_relative_angle += 360
        target_angle_error = (target_relative_angle - self.Best_Angle) / self.Max_Detection_Angle  # 0

        # 存储内容上一步内容
        self.Last_Target_Distance = target_relative_distance
        self.Last_Target_Distance_Error = target_distance_error
        self.Last_Target_Angle = target_relative_angle
        self.Last_Target_Angle_Error = target_angle_error
        self.Last_Tracker_Angle = tracker_angle
        self.Last_Seen_Target_Distance = self.Last_Target_Distance
        self.Last_Seen_Target_Angle = self.Last_Target_Angle
        # 更新渲染画面
        self.Canvas = utils.Get_Canvas(self.Target_Mode, self.Obstacle_Mode, self.Free_space, self.Blocking_Space,
                                       self.Target, self.Tracker, self.Static_Obstacles, self.Dynamic_Obstacles,
                                       self.Tracker_Trajectory, self.Target_Trajectory, self.Nav_Point,
                                       self.Last_Tracker_Angle)
        # 获取observation
        next_observation, self.Last_Seen_Target_Distance, self.Last_Seen_Target_Angle = utils.Get_Observation(
            self.Last_Target_Distance, self.Last_Target_Angle, self.Last_Seen_Target_Distance,
            self.Last_Seen_Target_Angle, self.Tracker, self.Angle, self.Static_Obstacles, self.Dynamic_Obstacles)
        # 需要的回合辅助信息
        info = {
            "total_step": self.Step_Count, "max_continuous_tracking_step": self.Max_Continuous_Tracking_Step,
            "success_flag": self.Success_Flag, "loss_flag": self.Loss_Flag, "collision_flag": self.CollisionFlag,
            "distance_error": self.Last_Target_Distance_Error, "angle_error": self.Last_Target_Angle_Error,
            "pos_tracker": self.Tracker, "pos_target": self.Target, "tracker_angle": self.Angle
        }

        return (next_observation, info)

    def render(self, mode='human'):
        """
        进行环境渲染
        :param mode:渲染模式
        :return None
        """
        # 设置窗口和时钟
        if self.Window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.Window = pygame.display.set_mode((self.Width, self.Height))  # 初始化一张画布
            pygame.display.set_caption("室外环境下动态目标跟踪")

        if self.Clock is None and mode == "human":
            self.Clock = pygame.time.Clock()
        # 获取画面
        self.Canvas = utils.Get_Canvas(self.Target_Mode, self.Obstacle_Mode, self.Free_space, self.Blocking_Space,
                                       self.Target, self.Tracker, self.Static_Obstacles, self.Dynamic_Obstacles,
                                       self.Tracker_Trajectory, self.Target_Trajectory, self.Nav_Point,
                                       self.Last_Tracker_Angle)
        # 根据渲染模式调整显示和储存
        if mode == "human":
            # The following line copies our drawing from 'canvas' to the visible window
            self.Window.blit(self.Canvas, self.Canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable
            self.Clock.tick(self.Metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.Canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        关闭当前环境
        :return:None
        """
        if self.Window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    '''主函数'''
    env = gym.make('TrackingEnv-v0', target_mode="Ram", obstacle_mode="Static", TEST=True)
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
