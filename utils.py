import numpy as np
import pygame
import math
import algo_config
import task_config
import utils


def reward_calculate(test, last_target_distance, last_target_angle, continuous_tracking_step, loss_step, max_continuous_tracking_step, tracker, target, static_obstacles, dynamic_obstacles):
    """
    计算当前动作对应的奖励，并且判断是否结束当前回合

    参数：
    - test: 表示当前是否处于测试模式。
    - last_target_distance: 上一次智能体与目标之间的距离。
    - last_target_angle: 上一次智能体与目标之间的夹角。
    - continuous_tracking_step: 连续跟踪步数。
    - loss_step: 丢失目标的步数。
    - max_continuous_tracking_step: 最大连续跟踪步数。
    - tracker: 智能体的位置和状态信息。
    - target: 目标的位置和状态信息。
    - static_obstacles: 静态障碍物列表。
    - dynamic_obstacles: 动态障碍物列表。

    返回值
    - reward: 单步奖励。
    - terminated: 是否因为碰撞或丢失目标而结束回合。
    - truncated: 是否结束回合（正常结束）。
    - success_flag: 是否成功跟踪目标。
    - collision_flag: 是否发生碰撞。
    - loss_flag: 是否丢失目标。
    - continuous_tracking_step: 连续跟踪步数。
    - loss_step: 丢失目标步数。
    - max_continuous_tracking_step: 最大连续跟踪步数。
    """
    # 初始化
    reward = 0
    terminated, truncated = False, False
    success_flag, collision_flag, loss_flag = False, False, False

    # case1: 与障碍物碰撞
    if utils.check_collision(tracker, target, static_obstacles, dynamic_obstacles):
        reward += task_config.collision_penalty
        truncated = True
        success_flag = False
        collision_flag = True
        loss_flag = False
        # print ("collision")

    # case2: 丢失目标
    elif last_target_distance > task_config.max_detection_distance or abs(last_target_angle) > task_config.max_detection_angle:
        if test:
            # 测试模式允许丢失一定长度
            continuous_tracking_step = 0 # 重置连续跟踪步数
            loss_step += 1 # 增加丢失目标步数
            if loss_step > task_config.max_loss_step: # 如果丢失步数超过最大限制
                reward += task_config.loss_penalty # 扣除丢失目标的惩罚分数
                truncated = True
                success_flag = False
                collision_flag = False
                loss_flag = True
            else:
                reward += -1  # 如果丢失步数未超过最大限制，稍作惩罚
        else:
            # 训练时不允许丢失
            continuous_tracking_step = 0
            reward += task_config.loss_penalty
            truncated = True
            success_flag = False
            collision_flag = False
            loss_flag = True
        # print ("lose the target")

    # 其他情况: 正常跟踪，根据情况分配奖励
    else:
        # 相关计数器处理
        continuous_tracking_step += 1 # 增加连续跟踪步数
        if continuous_tracking_step >= max_continuous_tracking_step: # 更新最大连续跟踪步数
            max_continuous_tracking_step = continuous_tracking_step
        loss_step = 0 # 重置丢失目标步数
        # reward计算
        step_tracking_reward_distance = - abs(
            last_target_distance - task_config.best_distance) / task_config.max_detection_distance
        step_tracking_reward_angle = - abs(last_target_angle) / task_config.max_detection_angle
        step_tracking_reward = 1 + step_tracking_reward_distance + step_tracking_reward_angle
        # clip
        if step_tracking_reward > 1:
            step_tracking_reward = 1
        elif step_tracking_reward < -1:
            step_tracking_reward = -1
        reward += 10 * step_tracking_reward
    return reward, terminated, truncated,success_flag,collision_flag,loss_flag,continuous_tracking_step,loss_step,max_continuous_tracking_step


def get_relative_elements(target, tracker, angle):
    """
    获取计算所需的辅助变量，包括目标与智能体之间的相对距离和角度。

    参数:
    - target: 目标的坐标，包含'x'和'y'键。
    - tracker: 智能体的坐标，包含'x'和'y'键。
    - angle: 智能体的当前朝向角度。

    返回:
    - target_relative_distance: 目标与智能体之间的相对距离。
    - target_distance_error: 目标与智能体之间距离的误差。
    - target_relative_angle: 目标与智能体之间的相对角度。
    - target_angle_error: 目标与智能体之间角度的误差。
    - tracker_angle: 智能体的当前朝向角度。
    """
    dx = target['x'] - tracker['x']
    dy = target['y'] - tracker['y']
    # 计算目标与智能体的相对距离及误差
    target_relative_distance = np.sqrt(dx ** 2 + dy ** 2)
    target_distance_error = (target_relative_distance - task_config.best_distance) / task_config.max_detection_distance  # 距离误差
    # 计算目标与智能体的相对角度及误差
    target_angle = np.degrees(np.arctan2(dy, dx)) % 360  # 范围：0-360
    tracker_angle = angle  # 范围：0-360
    target_relative_angle = target_angle - tracker_angle
    # 将相对角度转换为0-360度范围内的值
    if target_relative_angle > 180:
        target_relative_angle -= 360
    elif target_relative_angle < -180:
        target_relative_angle += 360
    target_angle_error = (target_relative_angle - task_config.best_angle) / task_config.max_detection_angle  # 角度误差
    return target_relative_distance, target_distance_error, target_relative_angle, target_angle_error, tracker_angle


def move_in_grid(agent, action, moving_size):
    """
    在全局坐标系下实现九宫格移动。

    参数:
    - agent: 移动对象，一个包含'x'和'y'键的字典，表示智能体的当前坐标。
    - action: 移动方向，整数编号： 0: 上, 1: 下, 2: 左, 3: 右, 4: 左上, 5: 右上, 6: 左下, 7: 右下, 8: 静止
    - moving_size: 每次移动的距离。
    返回:
    - agent: 移动后智能体的新坐标。
    """
    height = task_config.height
    width = task_config.width
    # 根据移动方向更新智能体的坐标
    if action == 0 and agent['y'] > 0:
        agent['y'] = max(0, agent['y'] - moving_size)
    elif action == 1 and agent['y'] < height - moving_size:
        agent['y'] = min(height - moving_size, agent['y'] + moving_size)
    elif action == 2 and agent['x'] > 0:
        agent['x'] = max(0, agent['x'] - moving_size)
    elif action == 3 and agent['x'] < width - moving_size:
        agent['x'] = min(width - moving_size, agent['x'] + moving_size)
    elif action == 4 and agent['x'] > 0 and agent['y'] > 0:
        agent['x'] = max(0, agent['x'] - moving_size)
        agent['y'] = max(0, agent['y'] - moving_size)
    elif action == 5 and agent['x'] < width - moving_size and agent['y'] > 0:
        agent['x'] = min(width - moving_size, agent['x'] + moving_size)
        agent['y'] = max(0, agent['y'] - moving_size)
    elif action == 6 and agent['x'] > 0 and agent['y'] < height - moving_size:
        agent['x'] = max(0, agent['x'] - moving_size)
        agent['y'] = min(height - moving_size, agent['y'] + moving_size)
    elif action == 7 and agent['x'] < width - moving_size and agent['y'] < height - moving_size:
        agent['x'] = min(width - moving_size, agent['x'] + moving_size)
        agent['y'] = min(height - moving_size, agent['y'] + moving_size)
    elif action == 8:
        agent['x'] = agent['x']
        agent['y'] = agent['y']
    return agent  # 返回更新后的智能体坐标


def target_nav(tracker, target, move_num, nav_point, static_obstacles, dynamic_obstacles, step_count):
    """
    使用人工势场（APF）算法实现导航类型目标的移动。

    参数:
    - tracker: 追踪者（智能体）的当前状态和位置。
    - target: 目标的当前状态和位置。
    - move_num: 目标已经移动的次数。
    - nav_point: 目标的导航点，即目标试图到达的位置。
    - static_obstacles: 环境中的静态障碍物列表。
    - dynamic_obstacles: 环境中的动态障碍物列表。
    - step_count: 当前算法运行的步数。

    返回:
    - target: 更新后的目标位置。
    - move_num: 更新后的移动步数。
    - nav_point: 更新后的导航点。
    """
    move_num += 1  # 增加移动次数
    # 计算目标受到的合力（引力和斥力）
    force_x, force_y = utils.calculate_force(target, nav_point, static_obstacles, dynamic_obstacles, a1=60)
    # 保存目标当前位置
    now_x = target['x']
    now_y = target['y']
    # 随机选择移动模式，主要是基于力的方向，但也引入一定随机性
    move_mode = np.random.choice(['by_force', 'random'], p=[0.8, 0.2])
    if move_mode == 'by_force':
        # 按照合力的方向移动
        new_x = target['x'] + np.sign(force_x) * task_config.moving_size
        new_y = target['y'] + np.sign(force_y) * task_config.moving_size
        # 检查新位置是否空闲，如果是，则移动到新位置
        if utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, new_x, new_y, False, False):
            target['x'] = new_x
            target['y'] = new_y
        # 如果新位置不空闲，尝试仅在x或y方向上移动
        elif utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, new_x, now_y, False, False):
            target['x'] = new_x
        elif utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, now_x, new_y, False, False):
            target['y'] = new_y
    else:
        # 引入随机性以尝试突破局部最优
        for _ in range(100):
            possible_moves = np.random.randint(0, 8)
            old_pos = {'x': target['x'], 'y': target['y']}
            new_pos = utils.move_in_grid(old_pos, possible_moves, task_config.moving_size)
            if utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, new_pos['x'], new_pos['y'], False, False):
                target['x'] = new_pos['x']
                target['y'] = new_pos['y']
                break
    # 如果目标到达导航点，则重新生成导航点
    if abs(target['x'] - nav_point['x']) <= task_config.pixel_size and abs(
            target['y'] - nav_point['y']) <= task_config.pixel_size:
        nav_point,move_num = utils.generate_nav_point(tracker, target, static_obstacles, dynamic_obstacles, step_count)
    # 如果长时间没有到达导航点，则也重新生成导航点
    if move_num % 20 == 0:
        nav_point,move_num = utils.generate_nav_point(tracker, target, static_obstacles, dynamic_obstacles, step_count)
    return target,move_num,nav_point


def calculate_force(target, nav_point, static_obstacles, dynamic_obstacles, a1):
    """
    计算目标在人工势场中的合力，包括来自导航点的引力和来自障碍物的斥力。

    参数:
    - target: 目标的当前位置，包含'x'和'y'键。
    - nav_point: 导航点的位置，目标试图到达的位置，包含'x'和'y'键。
    - static_obstacles: 环境中的静态障碍物列表。
    - dynamic_obstacles: 环境中的动态障碍物列表。
    - a1: 导航点对目标的引力系数。

    返回:
    - force_x: x方向上的合力。
    - force_y: y方向上的合力。
    """
    target_x, target_y = target['x'], target['y']  # 目标的当前坐标
    nav_x, nav_y = nav_point['x'], nav_point['y']  # 导航点的坐标
    force_x, force_y = 0, 0  # 初始化合力为0
    max_dist = 50  # 障碍物斥力作用的最大距离
    # 计算障碍物对目标的斥力
    for obstacle in static_obstacles + dynamic_obstacles:
        dist = np.sqrt((target_x - obstacle['x']) ** 2 + (target_y - obstacle['y']) ** 2)
        if dist != 0 and dist < max_dist:  # 如果距离不为0且在斥力作用范围内
            force_x += max(-0.05, (target_x - obstacle['x']) / dist ** 3)
            force_y += max(-0.05, (target_y - obstacle['y']) / dist ** 3)
    # 计算导航点对目标的引力
    dist = np.sqrt((target_x - nav_x) ** 2 + (target_y - nav_y) ** 2)
    if dist != 0:
        force_x += a1 * ((nav_x - target_x) / dist ** 3)
        force_y += a1 * ((nav_y - target_y) / dist ** 3)

    return force_x, force_y


def generate_nav_point(tracker, target, static_obstacles, dynamic_obstacles, step_count):
    """
    生成导航类型目标的导航点。

    参数:
    - tracker: 追踪者（智能体）的当前状态和位置。
    - target: 目标的当前状态和位置。
    - static_obstacles: 环境中的静态障碍物列表。
    - dynamic_obstacles: 环境中的动态障碍物列表。
    - step_count: 当前仿真的步数。

    返回:
    - nav_point: 新生成的导航点。
    - move_num: 移动次数，这里始终为0，因为导航点的生成不涉及移动。
    """
    move_num = 0  # 初始化移动次数计数器为0
    now_x = target['x']
    now_y = target['y']
    # 确定导航点生成的范围
    generate_range = 10 * task_config.pixel_size
    range_min_x = max(0, now_x - generate_range) // task_config.pixel_size
    range_max_x = min(now_x + generate_range, task_config.width) // task_config.pixel_size
    range_min_y = max(0, now_y - generate_range) // task_config.pixel_size
    range_max_y = min(now_y + generate_range, task_config.width) // task_config.pixel_size
    # 随机选取一个点作为导航点，并确保该点是可达的
    while True:
        nav_x = np.random.randint(range_min_x, range_max_x) * task_config.pixel_size
        nav_y = np.random.randint(range_min_y, range_max_y) * task_config.pixel_size
        if step_count == 0:  # 如果是仿真的第一步，则检查新导航点是否为自由空间
            if utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, nav_x, nav_y, False, True):
                nav_point = {'x': nav_x, 'y': nav_y}
                break
        else:  # 如果不是仿真的第一步，则检查新导航点是否为自由空间且不与障碍物重叠
            if utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, nav_x, nav_y, False, False):
                nav_point = {'x': nav_x, 'y': nav_y}
                break
    return nav_point,move_num


def dynamic_obstacle_move(dynamic_obstacles, moving_size, tracker, static_obstacles):
    """
    随机移动每个动态障碍物。

    参数:
    - dynamic_obstacles: 动态障碍物列表，每个障碍物都有'x'和'y'键表示其位置。
    - moving_size: 障碍物每次移动的步长。
    - tracker: 追踪者（智能体）的当前状态和位置。
    - static_obstacles: 环境中的静态障碍物列表。

    返回:
    - dynamic_obstacles: 更新位置后的动态障碍物列表。
    """
    for obstacle in dynamic_obstacles:
        # 随机决定障碍物是停止还是移动
        flag = np.random.choice(["stop", "move"], p=[0, 1])
        if flag == "move":
            for _ in range(100):
                # 尝试找到一个新的合法位置来移动障碍物
                possible_moves = np.random.randint(0, 8)  # 随机选择一个移动方向
                old_pos = {'x': obstacle['x'], 'y': obstacle['y']}  # 获取障碍物的当前位置
                new_pos = utils.move_in_grid(old_pos, possible_moves, 2 * moving_size)  # 根据随机方向和步长计算新位置
                # 检查新位置是否空闲且不是障碍物占据的位置
                if utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, new_pos['x'], new_pos['y'], is_static=False, is_initialize=False):
                    # 更新坐标
                    obstacle['x'] = new_pos['x']
                    obstacle['y'] = new_pos['y']
                    break # 找到合法的新位置后退出循环
        # 如果随机决定是'stop'，则障碍物保持不动，继续下一个迭代
        if flag == 'stop':
            continue
    return dynamic_obstacles


def move_clockwise(agent, direction, v):
    """
    使智能体在地图内沿顺时针或逆时针方向移动。

    参数:
    - agent: 移动对象，一个包含'x'和'y'键的字典，表示智能体的当前坐标。
    - direction: 移动方向，'clockwise'表示顺时针，'counterclockwise'表示逆时针。
    - v: 移动速度，即智能体每次移动的单位距离。

    返回:
    - agent: 更新位置后的智能体坐标。
    """
    x = agent['x']
    y = agent['y']
    flag = np.random.choice(["stop", "move"], p=[0, 1]) # 随机决定智能体是停止还是移动
    if flag == "move":  # 根据指定方向移动智能体
        if direction == "clockwise":  # 顺时针

            if x < 390 and y == 90:
                x += v
            elif x == 390 and y < 390:
                y += v
            elif x > 90 and y == 390:
                x -= v
            elif x == 90 and y > 90:
                y -= v
        else:  # 逆时针
            if x < 390 and y == 390:
                x += v
            elif x == 90 and y < 390:
                y += v
            elif x > 90 and y == 90:
                x -= v
            elif x == 390 and y > 90:
                y -= v
    # 更新智能体的位置
    agent['x'] = x
    agent['y'] = y
    return agent


def check_collision(tracker, target, static_obstacles, dynamic_obstacles):
    """
    检查智能体是否与障碍物或目标发生碰撞。

    参数:
    - tracker: 智能体的当前位置，包含'x'和'y'键。
    - target: 目标的当前位置，包含'x'和'y'键。
    - static_obstacles: 静态障碍物列表，每个障碍物都有'x'和'y'键表示其位置。
    - dynamic_obstacles: 动态障碍物列表，每个障碍物都有'x'和'y'键表示其位置。

    返回:
    - collide_flag: 布尔值，如果发生碰撞则为True，否则为False。
    """
    collide_flag = False  # 初始化碰撞标志为False
    # # 检查智能体是否与静态障碍物碰撞
    for obstacle in static_obstacles:
        if abs(obstacle['x'] - tracker['x']) <= task_config.pixel_size / 2 and abs(
                obstacle['y'] - tracker['y']) <= task_config.pixel_size / 2:
            collide_flag = True
    # 检查智能体是否与动态障碍物碰撞
    for obstacle in dynamic_obstacles:
        if abs(obstacle['x'] - tracker['x']) <= 3 and abs(obstacle['y'] - tracker['y']) <= 3:
            collide_flag = True
    # 检查智能体是否与目标碰撞
    if tracker['x'] == target['x'] and tracker['y'] == target['y']:
        collide_flag = True

    return collide_flag


def get_observation(last_target_distance, last_target_angle, last_seen_target_distance, last_seen_target_angle, tracker, angle, static_obstacles, dynamic_obstacles):
    """
    获取智能体的观测信息。

    参数:
    - last_target_distance: 上一次测量到的目标与智能体之间的距离。
    - last_target_angle: 上一次测量到的目标与智能体之间的相对角度。
    - last_seen_target_distance: 上一次看到目标时的距离。
    - last_seen_target_angle: 上一次看到目标时的相对角度。
    - tracker: 智能体的当前状态和位置。
    - angle: 智能体的当前朝向角度。
    - static_obstacles: 环境中的静态障碍物列表。
    - dynamic_obstacles: 环境中的动态障碍物列表。

    返回:
    - agent_observation: 智能体的观测信息，可能包括距离、角度和雷达数据。
    - last_seen_target_distance: 更新后的最后看到目标的距离。
    - last_seen_target_angle: 更新后的最后看到目标的角度。
    """
    # 检查目标是否在智能体的检测范围内
    if (last_target_distance <= task_config.max_detection_distance and abs(
            last_target_angle) <= task_config.max_detection_angle):
        # 如果目标在视野内，更新最后看到的状态，并使用当前的目标状态
        last_seen_target_distance = last_target_distance
        last_seen_target_angle = last_target_angle
        dist = last_target_distance
        agl = last_target_angle
    else:
        # 如果目标不在视野内，使用最后看到的状态
        dist = last_seen_target_distance
        agl = last_seen_target_angle
    # 归一化距离和角度
    distance = dist / task_config.max_detection_distance
    angle = agl / task_config.max_detection_angle
    # 如果启用了动作掩码，则获取掩码并将其与观测信息合并
    if task_config.mask_flag:
        action_mask = utils.get_action_mask(tracker, angle, static_obstacles, dynamic_obstacles)
        agent_observation = np.hstack((distance, angle, action_mask))
        return {"obs": agent_observation, "mask": action_mask},last_seen_target_distance,last_seen_target_angle
    else:  # 否则，获取雷达数据并将其与观测信息合并
        radar = utils.get_radar(tracker, angle, static_obstacles, dynamic_obstacles)
        agent_observation = np.hstack((distance, angle, radar))
        return agent_observation,last_seen_target_distance,last_seen_target_angle


def get_canvas(target_mode, obstacle_mode, free_space, blocking_space, target, tracker, static_obstacles, dynamic_obstacles, tracker_trajectory, target_trajectory, nav_point, last_tracker_angle):
    """
    绘制当前帧的渲染画面，包括智能体、目标、障碍物、轨迹等。

    参数:
    - target_mode: 目标的模式，例如"Nav"。
    - obstacle_mode: 障碍物的模式，例如"None"。
    - free_space: 表示自由空间的区域列表。
    - blocking_space: 表示阻塞空间的区域列表。
    - target: 目标的当前位置。
    - tracker: 智能体的当前位置。
    - static_obstacles: 静态障碍物列表。
    - dynamic_obstacles: 动态障碍物列表。
    - tracker_trajectory: 智能体的轨迹列表。
    - target_trajectory: 目标的轨迹列表。
    - nav_point: 导航点的位置。
    - last_tracker_angle: 智能体的最后朝向角度。

    返回:
    - canvas: 绘制好的pygame surface，表示当前帧的渲染画面。
    """
    # 加载并调整智能体和目标的图像尺寸
    tracker_img = task_config.tracker_img
    target_img = task_config.target_img
    tracker_img = pygame.transform.scale(tracker_img, (10, 10))
    target_img = pygame.transform.scale(target_img, (10, 10))
    # 创建一个新的pygame surface作为背景
    canvas = pygame.Surface((task_config.width, task_config.height))  # 创建一个新的pygame surface作为背景
    canvas.fill((255, 255, 255))  # 设置背景为白色
    # 如果没有障碍物，则将背景设置为绿色
    if obstacle_mode == 'None':
        canvas.fill((139, 195, 74))
    # 绘制自由空间和阻塞空间
    for space in free_space:
        pygame.draw.rect(canvas, (139, 195, 74),
                         pygame.Rect(space['x'], space['y'], task_config.pixel_size, task_config.pixel_size))  # 浅绿色正方形表示
    for block in blocking_space:
        pygame.draw.rect(canvas, (189, 189, 189),
                         pygame.Rect(block['x'], block['y'], task_config.pixel_size, task_config.pixel_size))  # 浅绿色正方形表示
    # 绘制静态和动态障碍物
    for obstacle in static_obstacles:
        pygame.draw.rect(canvas, (27, 94, 32),
                         pygame.Rect(obstacle['x'], obstacle['y'], task_config.pixel_size, task_config.pixel_size))
    for obstacle in dynamic_obstacles:
        pygame.draw.circle(canvas, (255, 235, 59),
                           (obstacle['x'] + task_config.pixel_size // 2, obstacle['y'] + task_config.pixel_size // 2),
                           3)
    # 绘制轨迹
    for i in range(len(tracker_trajectory) - 1):
        pygame.draw.line(canvas, (0, 0, 255), tracker_trajectory[i], tracker_trajectory[i + 1], 1)
    for i in range(len(target_trajectory) - 1):
        pygame.draw.line(canvas, (255, 0, 0), target_trajectory[i], target_trajectory[i + 1], 1)

    # 如果目标模式是"Nav"，则绘制导航点
    if target_mode == "Nav":
        pygame.draw.circle(canvas, (255, 0, 0),
                           (nav_point['x'] + task_config.pixel_size // 2, nav_point['y'] + task_config.pixel_size // 2),
                           task_config.pixel_size // 2)  # 导航点用红色圆形表示
    # 在指定位置绘制智能体和目标的图像
    canvas.blit(tracker_img,
                (tracker['x'] - tracker_img.get_rect().bottom / 2 + task_config.pixel_size // 2,
                 tracker['y'] - tracker_img.get_rect().right / 2 + task_config.pixel_size // 2))
    # 绘制智能体的可视区域（扇形）
    canvas.blit(target_img,
                (target['x'] - target_img.get_rect().bottom / 2 + task_config.pixel_size // 2,
                 target['y'] - target_img.get_rect().right / 2 + task_config.pixel_size // 2))
    utils.draw_sector(canvas, (0, 0, 255), tracker, last_tracker_angle)
    return canvas


def draw_sector(screen, color, tracker, last_tracker_angle, thickness=2):
    """
    在画面上绘制智能体的可视区域（扇形）。

    参数:
    - screen: 要在其上绘制扇形的pygame图层（screen）。
    - color: 扇形的边框颜色。
    - tracker: 智能体的当前位置，包含'x'和'y'键。
    - last_tracker_angle: 智能体的当前朝向角度。
    - thickness: 扇形边框的宽度，默认值为2。

    无返回值
    """
    # 计算扇形的中心点，即智能体的中心
    center = (tracker['x'] + task_config.pixel_size // 2, tracker['y'] + task_config.pixel_size // 2)
    axis_angle = last_tracker_angle  # 智能体的朝向角度作为扇形的对称轴方向
    sector_angle = task_config.max_detection_angle  # 扇形的张角，即智能体的视野角度
    radius = task_config.max_detection_distance  # 扇形的半径，即智能体的最大检测距离
    rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)  # 创建一个pygame.Rect对象，用于定义扇形的边界矩形
    # 计算扇形的开始角度和结束角度
    start_angle = -(axis_angle - sector_angle)
    end_angle = -(axis_angle + sector_angle)
    start_angle2 = axis_angle - sector_angle
    end_angle2 = axis_angle + sector_angle
    # 将角度转换为弧度，用于绘制扇形
    pygame.draw.arc(screen, color, rect, math.radians(end_angle), math.radians(start_angle), thickness)  # 绘制弧形
    # 计算扇形边界线的起始点和结束点
    start_pos = (center[0] + radius * math.cos(math.radians(start_angle2)),
                 center[1] + radius * math.sin(math.radians(start_angle2)))
    end_pos = (center[0] + radius * math.cos(math.radians(end_angle2)),
               center[1] + radius * math.sin(math.radians(end_angle2)))
    # 绘制扇形的两条边界线
    pygame.draw.line(screen, color, center, start_pos, thickness)  # 绘制两条边
    pygame.draw.line(screen, color, center, end_pos, thickness)


def get_radar(tracker, angle, static_obstacles, dynamic_obstacles):
    """
    模拟智能体的雷达系统，检测周围环境中的障碍物和边界。

    参数:
    - tracker: 智能体的当前位置，包含'x'和'y'键。
    - angle: 智能体的当前朝向角度。
    - static_obstacles: 静态障碍物列表，每个障碍物都有'x'和'y'键表示其位置。
    - dynamic_obstacles: 动态障碍物列表，每个障碍物都有'x'和'y'键表示其位置。

    返回:
    - to_avoid: 包含16个方向上障碍物距离的数组，用于表示智能体的雷达检测结果。
    """
    to_avoid = np.ones(16)  # 初始化一个长度为16的数组，每个元素初始化为1，表示探测范围
    # 对地图边缘的探测
    for i in range(16):
        # 计算当前方向的角度（从智能体的视角）
        angle_rad = math.radians(i * 22.5)  # 每个方向22.5度
        direction_x = math.cos(angle_rad)
        direction_y = math.sin(angle_rad)

        # 计算理论上的最大距离（不考虑碰撞）
        max_dist_x = task_config.max_detection_distance * direction_x
        max_dist_y = task_config.max_detection_distance * direction_y

        # 检测与地图边缘的距离
        dist_to_left_edge = max(0, tracker['x'] - 0)
        dist_to_right_edge = max(0, task_config.width - (tracker['x'] + max_dist_x))
        dist_to_top_edge = max(0, tracker['y'] - 0)
        dist_to_bottom_edge = max(0, task_config.height - (tracker['y'] + max_dist_y))

        # 更新雷达数据，取最近的边缘距离
        distance = min(dist_to_left_edge, dist_to_right_edge, dist_to_top_edge, dist_to_bottom_edge)
        if distance > task_config.max_detection_distance:
            continue
        to_avoid[i] = distance / task_config.max_detection_distance

    # 对障碍的探测
    for obstacle in static_obstacles + dynamic_obstacles:
        # 计算障碍物相对位置
        dx = obstacle['x'] - tracker['x']
        dy = obstacle['y'] - tracker['y']
        # 计算障碍物与智能体的距离
        dist = np.sqrt(dx ** 2 + dy ** 2) - task_config.pixel_size / 2
        if dist > task_config.max_detection_distance:
            continue
        # 计算障碍物与智能体的相对角度
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        base_index = int(angle / 22.5)
        base_angle = angle
        # 将角度映射到16个方向上
        basement = (base_index + 4) % 16  # 地图坐标系旋转
        direction_index = (basement - int(base_angle / 22.5)) % 16  # 相对角度
        # 更新雷达信息
        to_avoid[direction_index] = min(to_avoid[direction_index], dist / task_config.max_detection_distance)
    return to_avoid


def move(action):
    """
    根据智能体的动作决定其在局部坐标系下的移动距离和角度变化。

    参数:
    - action: 智能体的动作编号，整数类型。

    返回:
    - d: 移动距离。
    - alpha: 移动角度。
    """
    v = 5
    # 根据动作编号确定移动距离和角度
    if action == 2:
        d = v
        alpha = 0
    elif action == 0:
        d = v
        alpha = -45
    elif action == 1:
        d = v
        alpha = -22.5
    elif action == 3:
        d = v
        alpha = 22.5
    elif action == 4:
        d = v
        alpha = 45
    elif action == 7:
        d = -v
        alpha = 0
    elif action == 5:
        d = -v
        alpha = -45
    elif action == 6:
        d = -v
        alpha = -22.5
    elif action == 8:
        d = -v
        alpha = 22.5
    elif action == 9:
        d = -v
        alpha = 45
    else:
        d = 0
        alpha = 0
    return d, alpha


def map_process():
    """
    预置地图处理（现均为自由区域，若想划分区域可设置如下自定义内容）
    自由区域free_space：1-道路-灰色；2-活动场地-蓝色；3-草地-绿色
    障碍区域blocking_space:4-建筑物-棕色；5-固定障碍-橙色
    返回:
    - free_space: 包含地图上所有自由区域位置的列表。
    """
    free_space = []
    map_size = int(task_config.width / task_config.pixel_size)
    for row in range(0, map_size):
        for col in range(0, map_size):
            free_space.append({'x': row * task_config.pixel_size, 'y': col * task_config.pixel_size})
    return free_space


def is_free_space(tracker, static_obstacles, dynamic_obstacles, x, y, is_static=True, is_initialize=True):
    """
    检查给定的(x, y)位置是否空闲，即没有被障碍物占据，也没有超出地图边界。

    参数:
    - tracker: 智能体的当前位置，包含'x'和'y'键。
    - static_obstacles: 静态障碍物列表，每个障碍物都有'x'和'y'键表示其位置。
    - dynamic_obstacles: 动态障碍物列表，每个障碍物都有'x'和'y'键表示其位置。
    - x: 要检查的坐标的x值。
    - y: 要检查的坐标的y值。
    - is_static: 标志，指示是否考虑静态障碍物，默认为True。
    - is_initialize: 初始化状态标志，用于在重置环境时使用，默认为True。

    返回:
    - 布尔值：如果(x, y)位置空闲，则返回True；否则返回False。
    """
    # 检查是否出界
    if x < 0 or x >= task_config.width or y < 0 or y >= task_config.height:
        return False
    # 检查是否与静态障碍物重叠
    for obstacle in static_obstacles:
        if abs(obstacle['x'] - x) <= task_config.pixel_size / 2 and abs(obstacle['y'] - y) <= task_config.pixel_size / 2:
            return False
    # 如果不是初始化状态，检查智能体是否与目标位置重叠
    if not is_initialize:
        if tracker['x'] == x and tracker['y'] == y:
            return False
    # 如果需要考虑动态障碍物
    if not is_static:
        for obstacle in dynamic_obstacles:
            if abs(obstacle['x'] - x) <= 3 and abs(obstacle['y'] - y) <= 3:
                return False
    # 如果没有障碍物重叠，返回True
    return True


def get_action_mask(tracker, angle, static_obstacles, dynamic_obstacles):
    """
    根据智能体的雷达信息生成动作掩码，用于限制智能体的动作，以避免碰撞。

    参数:
    - tracker: 智能体的当前位置和状态。
    - angle: 智能体的当前朝向角度。
    - static_obstacles: 环境中的静态障碍物列表。
    - dynamic_obstacles: 环境中的动态障碍物列表。

    返回:
    - action_mask: 动作掩码数组，用于指示哪些动作是不合法的（掩码）。
    """
    action_mask = np.ones(11)  # 初始化动作掩码为1，表示所有动作都是合法的
    mask_dist = 0.3  # 设置障碍物检测的距离阈值
    # 获取雷达信息，检测智能体周围的障碍物
    radar = utils.get_radar(tracker, angle, static_obstacles, dynamic_obstacles)
    mask_flag = task_config.mask_flag
    # 如果启用了掩码标志，则根据雷达信息更新动作掩码
    if mask_flag:
        # 根据雷达数据设置动作掩码，如果某个方向上检测到障碍物，则禁止该方向的动作
        if radar[1] <= mask_dist or radar[2] <= mask_dist:
            action_mask[0] = 0
        if radar[2] <= mask_dist or radar[3] <= mask_dist:
            action_mask[1] = 0
        if radar[3] <= mask_dist or radar[4] <= mask_dist:
            action_mask[2] = 0
        if radar[4] <= mask_dist or radar[5] <= mask_dist:
            action_mask[3] = 0
        if radar[5] <= mask_dist or radar[6] <= mask_dist:
            action_mask[4] = 0
        if radar[9] <= mask_dist or radar[10] <= mask_dist:
            action_mask[5] = 0
        if radar[10] <= mask_dist or radar[11] <= mask_dist:
            action_mask[6] = 0
        if radar[11] <= mask_dist or radar[12] <= mask_dist:
            action_mask[7] = 0
        if radar[12] <= mask_dist or radar[13] <= mask_dist:
            action_mask[8] = 0
        if radar[14] <= mask_dist or radar[13] <= mask_dist:
            action_mask[9] = 0
    return action_mask  # 返回生成的动作掩码


def generate_static_obstacles(tracker, dynamic_obstacles, blocking_space):
    """
    在仿真环境中随机均匀地生成静态障碍物。

    参数:
    - tracker: 智能体的当前位置和状态。
    - dynamic_obstacles: 动态障碍物列表。
    - blocking_space: 阻塞空间列表，表示不可通过的区域。

    返回:
    - static_obstacles: 生成的静态障碍物列表。
    """
    static_obstacles_density = 2  # 每个区域内障碍物数量
    static_obstacles = []
    obstacle_positions = utils.generate_random_positions(static_obstacles_density)  # 生成随机位置

    # 根据位置随机生成障碍物形状和大小
    for position in obstacle_positions:
        # 随机选择障碍物的形状
        shape = np.random.choice(['rectangle', 'cross', 'circle', 'parallel_line', 'vertical_line'],
                                 p=task_config.obstacle_kind_probability)
        # 根据选择的形状生成障碍物，并添加到静态障碍物列表中
        if shape == 'rectangle':
            obstacle = utils.generate_rectangle(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        elif shape == 'cross':
            obstacle = utils.generate_cross(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        elif shape == 'circle':
            obstacle = utils.generate_circle(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        elif shape == 'parallel_line':
            obstacle = utils.generate_parallel_line(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        else:
            obstacle = utils.generate_vertical_line(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        static_obstacles.extend(obstacle)
    return static_obstacles

def generate_static_obstacles_randomly(tracker, dynamic_obstacles, blocking_space):
    """
    在仿真环境中随机非均匀地生成静态障碍物。

    参数:
    - tracker: 智能体的当前位置和状态。
    - dynamic_obstacles: 动态障碍物列表。
    - blocking_space: 阻塞空间列表，表示不可通过的区域。

    返回:
    - static_obstacles: 生成的静态障碍物列表。
    """
    num_positions = 50  # 地图内障碍物数量
    static_obstacles = []
    positions = []
    # 生成随机位置
    for _ in range(num_positions):
        a = np.random.randint(0, 49)
        b = np.random.randint(0, 49)
        x = a * task_config.pixel_size
        y = b * task_config.pixel_size
        if not ((x, y) in positions):
            positions.append((x, y))
    # 根据随机生成的位置，为每个位置随机生成障碍物的形状和大小
    for position in positions:
        # 随机选择障碍物的形状
        shape = np.random.choice(['rectangle', 'cross', 'circle', 'parallel_line', 'vertical_line'],
                                 p=[0.4, 0.1, 0.1, 0.2, 0.2])
        # 根据选择的形状生成障碍物，并添加到静态障碍物列表中
        if shape == 'rectangle':
            obstacle = utils.generate_rectangle(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        elif shape == 'cross':
            obstacle = utils.generate_cross(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        elif shape == 'circle':
            obstacle = utils.generate_circle(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        elif shape == 'parallel_line':
            obstacle = utils.generate_parallel_line(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        else:
            obstacle = utils.generate_vertical_line(position, tracker, static_obstacles, dynamic_obstacles, blocking_space)
        static_obstacles.extend(obstacle)
    return static_obstacles


def generate_random_positions(num_positions):
    """
    在地图的每个区域中生成指定数量的随机位置坐标。

    参数:
    - num_positions: 每个区域中需要生成的位置数量。

    返回:
    - positions: 包含所有生成的随机位置坐标的列表。
    """
    positions = []  # 初始化一个空列表，用于存储生成的位置坐标
    area_dist = []  # 初始化一个空列表，用于存储地图的每个区域的中心点
    # 将地图划分为25个区域，并计算每个区域的中心点
    for i in range(int(task_config.width / 100)):
        for j in range(int(task_config.width / 100)):
            area_dist.append({'x': i, 'y': j})
    # 对于每个区域，生成指定数量的随机位置
    for area in area_dist:
        for _ in range(num_positions):
            # 在每个区域内部生成随机坐标
            a = np.random.randint(0, 9)
            b = np.random.randint(0, 9)
            x = (task_config.pixel_size * area['x'] + a) * task_config.pixel_size
            y = (task_config.pixel_size * area['y'] + b) * task_config.pixel_size
            # 确保不重复添加相同的坐标
            if not ((x, y) in positions):
                positions.append((x, y))
    return positions  # 返回包含所有生成的随机位置坐标的列表


def generate_rectangle(position, tracker, static_obstacles, dynamic_obstacles, blocking_space):
    """
    在给定的中心位置生成一个矩形障碍物。

    参数:
    - position: 矩形障碍物的中心位置，包含'x'和'y'键。
    - tracker: 智能体的当前位置和状态。
    - static_obstacles: 静态障碍物列表。
    - dynamic_obstacles: 动态障碍物列表。
    - blocking_space: 阻塞空间列表，表示不可通过的区域。

    返回:
    - obstacle: 包含矩形障碍物所有覆盖坐标点的列表。
    """
    height = np.random.randint(task_config.min_obstacle_size // 2, task_config.max_obstacle_size // 2)
    width = np.random.randint(task_config.min_obstacle_size // 2, task_config.max_obstacle_size // 2)
    x, y = position
    obstacle = []  # 初始化障碍物坐标列表
    # 遍历矩形区域内的每个点
    for i in range(x - height // 2 * task_config.pixel_size, x + height // 2 * task_config.pixel_size, task_config.pixel_size):
        for j in range(y - width // 2 * task_config.pixel_size, y + width // 2 * task_config.pixel_size, task_config.pixel_size):
            # 检查该点是否不在阻塞空间内
            if not({'x':i,'y':j} in blocking_space):
                # 检查该点是否在地图范围内且是空闲的
                if 0 <= i < task_config.width and 0 <= j < task_config.height and utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, i, j, is_static=True, is_initialize=True):
                    obstacle.append({"x": i, "y": j})  # 添加到障碍物坐标列表
    return obstacle


def generate_cross(position, tracker, static_obstacles, dynamic_obstacles, blocking_space):
    """
    在给定的中心位置生成一个十字形障碍物。

    参数:
    - position: 十字形障碍物的中心位置，包含'x'和'y'键。
    - tracker: 智能体的当前位置和状态。
    - static_obstacles: 静态障碍物列表。
    - dynamic_obstacles: 动态障碍物列表。
    - blocking_space: 阻塞空间列表，表示不可通过的区域。

    返回:
    - obstacle: 包含十字形障碍物所有覆盖坐标点的列表。
    """
    size = np.random.randint(task_config.min_obstacle_size, task_config.max_obstacle_size)
    x, y = position
    obstacle = []
    # 遍历十字形的横线部分
    for i in range(x - size // 2 * task_config.pixel_size, x + size // 2 * task_config.pixel_size, task_config.pixel_size):
        if 0 <= i < task_config.width and utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, i, y, is_static=True, is_initialize=True) and not({'x':i, 'y':y} in blocking_space):
            obstacle.append({"x": i, "y": y})
    # 遍历十字形的竖线部分
    for j in range(y - size // 2 * task_config.pixel_size, y + size // 2 * task_config.pixel_size, task_config.pixel_size):
        if 0 <= j < task_config.height and utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, x, j, is_static=True, is_initialize=True) and not({'x':x, 'y':j} in blocking_space):
            obstacle.append({"x": x, "y": j})
    return obstacle


def generate_circle(position, tracker, static_obstacles, dynamic_obstacles, blocking_space):
    """
    在给定的中心位置生成一个圆形障碍物。

    参数:
    - position: 圆形障碍物的中心位置，包含'x'和'y'键。
    - tracker: 智能体的当前位置和状态。
    - static_obstacles: 静态障碍物列表。
    - dynamic_obstacles: 动态障碍物列表。
    - blocking_space: 阻塞空间列表，表示不可通过的区域。

    返回:
    - obstacle: 包含圆形障碍物所有覆盖坐标点的列表。
    """
    radius = np.random.randint(task_config.min_obstacle_size // 2, task_config.max_obstacle_size // 2)
    x, y = position
    obstacle = []
    # 遍历圆形区域内的每个点
    for i in range(x - radius * task_config.pixel_size, x + (radius + 1) * task_config.pixel_size,task_config.pixel_size):
        for j in range(y - radius * task_config.pixel_size, y + (radius + 1) * task_config.pixel_size, task_config.pixel_size):
            # 检查该点是否在地图范围内，是否在圆形区域内，是否是空闲的，并且不在阻塞空间内
            if 0 <= i < task_config.width and 0 <= j < task_config.height and (x - i) ** 2 + (y - j) ** 2 <= (
                    radius * task_config.pixel_size) ** 2 and utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, i, j, is_static=True, is_initialize=True) and not({'x':i, 'y':j} in blocking_space):
                obstacle.append({"x": i, "y": j})
    return obstacle


def generate_parallel_line(position, tracker, static_obstacles, dynamic_obstacles, blocking_space):
    """
    在给定的中心位置生成一个横向条形障碍物。

    参数:
    - position: 横向条形障碍物的中心位置，包含'x'和'y'键。
    - tracker: 智能体的当前位置和状态。
    - static_obstacles: 静态障碍物列表。
    - dynamic_obstacles: 动态障碍物列表。
    - blocking_space: 阻塞空间列表，表示不可通过的区域。

    返回:
    - obstacle: 包含横向条形障碍物所有覆盖坐标点的列表。
    """
    length = np.random.randint(task_config.min_obstacle_size, task_config.max_obstacle_size)
    x, y = position
    obstacle = []
    # 遍历横向条形障碍物的长度范围
    for i in range(x - length // 2 * task_config.pixel_size, x + length // 2 * task_config.pixel_size, task_config.pixel_size):
        # 检查该点是否在地图范围内，是否是空闲的，并且不在阻塞空间内
        if 0 <= i < task_config.width and utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, i, y, is_static=True, is_initialize=True) and not({'x':i, 'y':y} in blocking_space):
            obstacle.append({"x": i, "y": y})
    return obstacle


def generate_vertical_line(position, tracker, static_obstacles, dynamic_obstacles, blocking_space):
    """
    在给定的中心位置生成一个纵向条形障碍物。

    参数:
    - position: 横向条形障碍物的中心位置，包含'x'和'y'键。
    - tracker: 智能体的当前位置和状态。
    - static_obstacles: 静态障碍物列表。
    - dynamic_obstacles: 动态障碍物列表。
    - blocking_space: 阻塞空间列表，表示不可通过的区域。

    返回:
    - obstacle: 包含纵向条形障碍物所有覆盖坐标点的列表。
    """
    length = np.random.randint(task_config.min_obstacle_size, task_config.max_obstacle_size)
    x, y = position
    obstacle = []
    # 遍历纵向条形障碍物的长度范围
    for i in range(y - length // 2 * task_config.pixel_size, y + length // 2 * task_config.pixel_size, task_config.pixel_size):
        # 检查该点是否在地图范围内，是否是空闲的，并且不在阻塞空间内
        if 0 <= i < task_config.height and utils.is_free_space(tracker, static_obstacles, dynamic_obstacles, x, i, is_static=True, is_initialize=True) and not({'x':x, 'y':i} in blocking_space):
            obstacle.append({"x": x, "y": i})
    return obstacle

