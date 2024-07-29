import numpy as np
import pygame
import math
import config
import utils


def Reward_Calculate(TEST,Last_Target_Distance,Last_Target_Angle,Continuous_Tracking_Step,Loss_Step,Max_Continuous_Tracking_Step,Tracker,Target,Static_Obstacles,Dynamic_Obstacles):
    '''
    计算当前动作对应的奖励，并且判断是否结束当前回合
    :return: reward单步奖励，terminated, truncated回合结束标志
    '''
    reward = 0
    terminated, truncated = False, False
    Success_Flag, CollisionFlag, Loss_Flag = False,False,False

    # case1: 与障碍物碰撞
    if utils.Check_Collision(Tracker,Target,Static_Obstacles,Dynamic_Obstacles):
        reward += config.collision_penalty
        terminated = True
        Success_Flag = False
        CollisionFlag = True
        Loss_Flag = False
        # print ("collision")

    # case2: 丢失目标
    elif Last_Target_Distance > config.max_detection_distance or abs(Last_Target_Angle) > config.max_detection_angle:
        if TEST:
            # 测试模式允许丢失一定长度
            Continuous_Tracking_Step = 0
            Loss_Step += 1
            if Loss_Step > config.max_loss_step:
                reward += config.loss_penalty
                terminated = True
                Success_Flag = False
                CollisionFlag = False
                Loss_Flag = True
        else:
            # 训练时不允许丢失
            Continuous_Tracking_Step = 0
            reward += config.loss_penalty
            terminated = True
            Success_Flag = False
            CollisionFlag = False
            Loss_Flag = True
        # print ("lose the target")

    # 其他情况: 正常跟踪，根据情况分配奖励
    else:
        # 相关计数器处理
        Continuous_Tracking_Step += 1
        if Continuous_Tracking_Step >= Max_Continuous_Tracking_Step:
            Max_Continuous_Tracking_Step = Continuous_Tracking_Step
        Loss_Step = 0
        # reward计算
        step_tracking_reward_distance = - abs(
            Last_Target_Distance - config.best_distance) / config.max_detection_distance
        step_tracking_reward_angle = - abs(Last_Target_Angle) / config.max_detection_angle
        step_tracking_reward = 1 + step_tracking_reward_distance + step_tracking_reward_angle
        # clip
        if step_tracking_reward > 1:
            step_tracking_reward = 1
        elif step_tracking_reward < -1:
            step_tracking_reward = -1
        reward += 10 * step_tracking_reward
    return reward, terminated, truncated,Success_Flag,CollisionFlag,Loss_Flag,Continuous_Tracking_Step,Loss_Step,Max_Continuous_Tracking_Step


def Get_Relative_Elements(Target,Tracker,Angle):
    '''
    获取计算所需的辅助变量
    :return: 相对距离，距离误差，相对角度，角度误差，tracker当前朝向
    '''
    dx = Target['x'] - Tracker['x']
    dy = Target['y'] - Tracker['y']
    # 计算目标与智能体的相对距离
    target_relative_distance = np.sqrt(dx ** 2 + dy ** 2)
    target_distance_error = (target_relative_distance - config.best_distance) / config.max_detection_distance  # 距离误差
    # 计算目标与智能体的相对角度
    target_angle = np.degrees(np.arctan2(dy, dx)) % 360  # 范围：0-360
    tracker_angle = Angle  # 范围：0-360
    target_relative_angle = target_angle - tracker_angle
    if target_relative_angle > 180:
        target_relative_angle -= 360
    elif target_relative_angle < -180:
        target_relative_angle += 360
    target_angle_error = (target_relative_angle - config.best_angle) / config.max_detection_angle  # 角度误差
    return target_relative_distance, target_distance_error, target_relative_angle, target_angle_error, tracker_angle


def Move_In_Grid(agent, action, moving_size):
    '''
    实现全局坐标系下的九宫格移动
    :param agent:移动对象
    :param action: 移动方向 0: 上, 1: 下, 2: 左, 3: 右, 4: 左上, 5: 右上, 6: 左下, 7: 右下, 8: 静止
    :param moving_size: 移动距离
    :return: 移动对象新坐标
    '''
    if action == 0 and agent['y'] > 0:
        agent['y'] = max(0, agent['y'] - moving_size)
    elif action == 1 and agent['y'] < config.height - moving_size:
        agent['y'] = min(config.height - moving_size, agent['y'] + moving_size)
    elif action == 2 and agent['x'] > 0:
        agent['x'] = max(0, agent['x'] - moving_size)
    elif action == 3 and agent['x'] < config.width - moving_size:
        agent['x'] = min(config.width - moving_size, agent['x'] + moving_size)
    elif action == 4 and agent['x'] > 0 and agent['y'] > 0:
        agent['x'] = max(0, agent['x'] - moving_size)
        agent['y'] = max(0, agent['y'] - moving_size)
    elif action == 5 and agent['x'] < config.width - moving_size and agent['y'] > 0:
        agent['x'] = min(config.width - moving_size, agent['x'] + moving_size)
        agent['y'] = max(0, agent['y'] - moving_size)
    elif action == 6 and agent['x'] > 0 and agent['y'] < config.height - moving_size:
        agent['x'] = max(0, agent['x'] - moving_size)
        agent['y'] = min(config.height - moving_size, agent['y'] + moving_size)
    elif action == 7 and agent['x'] < config.width - moving_size and agent['y'] < config.height - moving_size:
        agent['x'] = min(config.width - moving_size, agent['x'] + moving_size)
        agent['y'] = min(config.height - moving_size, agent['y'] + moving_size)
    elif action == 8:
        agent['x'] = agent['x']
        agent['y'] = agent['y']
    return agent


def Target_Nav(Tracker,Target,Move_Num,Nav_Point,Static_Obstacles,Dynamic_Obstacles,Step_Count):
    '''
    以APF实现Nav类型目标的移动
    :return: None
    '''
    Move_Num += 1
    # 计算引斥力
    force_x, force_y = utils.Calculate_Force(Target,Nav_Point,Static_Obstacles,Dynamic_Obstacles,a1=60)
    now_x = Target['x']
    now_y = Target['y']
    move_mode = np.random.choice(['by_force', 'random'], p=[0.8, 0.2])
    if move_mode == 'by_force':
        # 沿APF指导移动
        new_x = Target['x'] + np.sign(force_x) * config.moving_size
        new_y = Target['y'] + np.sign(force_y) * config.moving_size
        if utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,new_x, new_y, False, False):
            Target['x'] = new_x
            Target['y'] = new_y
        elif utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,new_x, now_y, False, False):
            Target['x'] = new_x
        elif utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,now_x, new_y, False, False):
            Target['y'] = new_y
    else:
        # 引入随机性突破局部最优
        for _ in range(100):
            possible_moves = np.random.randint(0, 8)
            old_pos = {'x': Target['x'], 'y': Target['y']}
            new_pos = utils.Move_In_Grid(old_pos, possible_moves, config.moving_size)
            if utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,new_pos['x'], new_pos['y'], False, False):
                Target['x'] = new_pos['x']
                Target['y'] = new_pos['y']
                break
    # 到达导航点则重新生成导航点
    if abs(Target['x'] - Nav_Point['x']) <= config.pixel_size and abs(
            Target['y'] - Nav_Point['y']) <= config.pixel_size:
        Nav_Point,Move_Num = utils.Generate_Nav_Point(Tracker,Target,Static_Obstacles,Dynamic_Obstacles,Step_Count)
    # 太长时间没有到达导航点则重新生成导航点
    if Move_Num % 20 == 0:
        Nav_Point,Move_Num = utils.Generate_Nav_Point(Tracker,Target,Static_Obstacles,Dynamic_Obstacles,Step_Count)
    return Target,Move_Num,Nav_Point

def Calculate_Force(Target,Nav_Point,Static_Obstacles,Dynamic_Obstacles,a1):
    '''
    计算人工势场合力
    :param a1:引力系数
    :return: force_x,force_y
    '''
    target_x, target_y = Target['x'], Target['y']
    nav_x, nav_y = Nav_Point['x'], Nav_Point['y']
    force_x, force_y = 0, 0
    max_dist = 50
    # 计算障碍物对目标的斥力
    for obstacle in Static_Obstacles + Dynamic_Obstacles:
        dist = np.sqrt((target_x - obstacle['x']) ** 2 + (target_y - obstacle['y']) ** 2)
        if dist != 0 and dist < max_dist:
            force_x += max(-0.05, (target_x - obstacle['x']) / dist ** 3)
            force_y += max(-0.05, (target_y - obstacle['y']) / dist ** 3)
    # 计算导航点对目标的引力
    dist = np.sqrt((target_x - nav_x) ** 2 + (target_y - nav_y) ** 2)
    if dist != 0:
        force_x += a1 * ((nav_x - target_x) / dist ** 3)
        force_y += a1 * ((nav_y - target_y) / dist ** 3)

    return force_x, force_y


def Generate_Nav_Point(Tracker,Target,Static_Obstacles,Dynamic_Obstacles,Step_Count):
    '''
    生成Nav目标导航点
    :return: nav_point
    '''
    Move_Num = 0  # 清空计数器
    now_x = Target['x']
    now_y = Target['y']
    # 确定导航点生成范围
    generate_range = 10 * config.pixel_size
    range_min_x = max(0, now_x - generate_range) // config.pixel_size
    range_max_x = min(now_x + generate_range, config.width) // config.pixel_size
    range_min_y = max(0, now_y - generate_range) // config.pixel_size
    range_max_y = min(now_y + generate_range, config.width) // config.pixel_size
    # 随机选取并保证可达
    while True:
        nav_x = np.random.randint(range_min_x, range_max_x) * config.pixel_size
        nav_y = np.random.randint(range_min_y, range_max_y) * config.pixel_size

        if Step_Count == 0:
            if utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,nav_x, nav_y, False, True):
                nav_point = {'x': nav_x, 'y': nav_y}
                break
        else:
            if utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,nav_x, nav_y, False, False):
                nav_point = {'x': nav_x, 'y': nav_y}
                break
    return nav_point,Move_Num


def Dynamic_Obstacle_Move(Dynamic_Obstacles,Moving_Size,Tracker,Static_Obstacles):
    '''
    实现随机动态障碍移动
    :return:
    '''
    for obstacle in Dynamic_Obstacles:
        flag = np.random.choice(["stop", "move"], p=[0, 1])
        if flag == "move":
            for _ in range(100):
                possible_moves = np.random.randint(0, 8)
                old_pos = {'x': obstacle['x'], 'y': obstacle['y']}
                new_pos = utils.Move_In_Grid(old_pos, possible_moves, 2 * Moving_Size)
                if utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,new_pos['x'], new_pos['y'], is_static=False, is_initialize=False):
                    obstacle['x'] = new_pos['x']
                    obstacle['y'] = new_pos['y']
                    break
        if flag == 'stop':
            continue
    return Dynamic_Obstacles

def Move_Clockwise(agent, direction, v):
    '''
    实现地图内的顺时针和逆时针运动
    :param agent:移动对象
    :param direction: 移动方向
    :param v: 移动速度
    :return: None
    '''
    x = agent['x']
    y = agent['y']
    flag = np.random.choice(["stop", "move"], p=[0, 1])
    if flag == "move":
        if direction == "clockwise":
            # 顺时针

            if x < 390 and y == 90:
                x += v
            elif x == 390 and y < 390:
                y += v
            elif x > 90 and y == 390:
                x -= v
            elif x == 90 and y > 90:
                y -= v
        else:
            # 逆时针
            if x < 390 and y == 390:
                x += v
            elif x == 90 and y < 390:
                y += v
            elif x > 90 and y == 90:
                x -= v
            elif x == 390 and y > 90:
                y -= v
    agent['x'] = x
    agent['y'] = y
    return agent

def Check_Collision(Tracker,Target,Static_Obstacles,Dynamic_Obstacles):
    '''
    判断智能体是否发生碰撞
    :return:bool collide_flag
    '''
    collide_flag = False
    # 检查是否与障碍物碰撞
    for obstacle in Static_Obstacles:
        if abs(obstacle['x'] - Tracker['x']) <= config.pixel_size / 2 and abs(
                obstacle['y'] - Tracker['y']) <= config.pixel_size / 2:
            collide_flag = True
    for obstacle in Dynamic_Obstacles:
        if abs(obstacle['x'] - Tracker['x']) <= 3 and abs(obstacle['y'] - Tracker['y']) <= 3:
            collide_flag = True
    # 检查是否与目标碰撞
    if Tracker['x'] == Target['x'] and Tracker['y'] == Target['y']:
        collide_flag = True

    return collide_flag


def Get_Observation(Last_Target_Distance,Last_Target_Angle,Last_Seen_Target_Distance,Last_Seen_Target_Angle,Tracker,Angle,Static_Obstacles,Dynamic_Obstacles):
    if (Last_Target_Distance <= config.max_detection_distance and abs(
            Last_Target_Angle) <= config.max_detection_angle):
        # 如果目标在视野内，更新最后看到的状态，并使用当前的目标状态
        Last_Seen_Target_Distance = Last_Target_Distance
        Last_Seen_Target_Angle = Last_Target_Angle
        dist = Last_Target_Distance
        agl = Last_Target_Angle
    else:
        # 如果目标不在视野内，使用最后看到的状态
        dist = Last_Seen_Target_Distance
        agl = Last_Seen_Target_Angle

    distance = dist / config.max_detection_distance
    angle = agl / config.max_detection_angle
    if config.mask_flag:
        action_mask = utils.Get_Action_Mask(Tracker,Angle,Static_Obstacles,Dynamic_Obstacles)
        agent_observation = np.hstack((distance, angle, action_mask))
        return {"obs": agent_observation, "mask": action_mask},Last_Seen_Target_Distance,Last_Seen_Target_Angle
    else:
        radar = utils.Get_Radar(Tracker,Angle,Static_Obstacles,Dynamic_Obstacles)
        agent_observation = np.hstack((distance, angle, radar))
        return agent_observation,Last_Seen_Target_Distance,Last_Seen_Target_Angle


def Get_Canvas(Target_Mode,Obstacle_Mode,Free_space,Blocking_Space,Target,Tracker,Static_Obstacles,Dynamic_Obstacles,Tracker_Trajectory,Target_Trajectory,Nav_Point,Last_Tracker_Angle):
    """
    绘制当前帧渲染画面
    """
    Tracker_Img = config.tracker_img
    Target_Img = config.target_img
    # 改变尺寸
    Tracker_Img = pygame.transform.scale(Tracker_Img, (10, 10))
    Target_Img = pygame.transform.scale(Target_Img, (10, 10))

    Canvas = pygame.Surface((config.width, config.height))  # 创建一个新的pygame surface作为背景
    Canvas.fill((255, 255, 255))  # 设置背景为白色
    if Obstacle_Mode == 'None':
        Canvas.fill((139, 195, 74))  # 设置背景为绿色
    for space in Free_space:
        pygame.draw.rect(Canvas, (139, 195, 74),
                         pygame.Rect(space['x'], space['y'], config.pixel_size, config.pixel_size))  # 浅绿色正方形表示
    for block in Blocking_Space:
        pygame.draw.rect(Canvas, (189, 189, 189),
                         pygame.Rect(block['x'], block['y'], config.pixel_size, config.pixel_size))  # 浅绿色正方形表示

    for obstacle in Static_Obstacles:
        pygame.draw.rect(Canvas, (27, 94, 32),
                         pygame.Rect(obstacle['x'], obstacle['y'], config.pixel_size, config.pixel_size))
    for obstacle in Dynamic_Obstacles:
        pygame.draw.circle(Canvas, (255, 235, 59),
                           (obstacle['x'] + config.pixel_size // 2, obstacle['y'] + config.pixel_size // 2),
                           3)
    # 绘制轨迹
    for i in range(len(Tracker_Trajectory) - 1):
        pygame.draw.line(Canvas, (0, 0, 255), Tracker_Trajectory[i], Tracker_Trajectory[i + 1], 1)
    for i in range(len(Target_Trajectory) - 1):
        pygame.draw.line(Canvas, (255, 0, 0), Target_Trajectory[i], Target_Trajectory[i + 1], 1)

    # pygame.draw.circle(self.canvas, (0, 0, 255),(self.tracker['x'] + self.pixel_size // 2, self.tracker['y'] + self.pixel_size // 2),50, 2)
    if Target_Mode == "Nav":
        pygame.draw.circle(Canvas, (255, 0, 0),
                           (Nav_Point['x'] + config.pixel_size // 2, Nav_Point['y'] + config.pixel_size // 2),
                           config.pixel_size // 2)  # 导航点用红色圆形表示

    Canvas.blit(Tracker_Img,
                     (Tracker['x'] - Tracker_Img.get_rect().bottom / 2 + config.pixel_size // 2,
                      Tracker['y'] - Tracker_Img.get_rect().right / 2 + config.pixel_size // 2))

    Canvas.blit(Target_Img,
                     (Target['x'] - Target_Img.get_rect().bottom / 2 + config.pixel_size // 2,
                      Target['y'] - Target_Img.get_rect().right / 2 + config.pixel_size // 2))
    utils.Draw_Sector(Canvas, (0, 0, 255),Tracker,Last_Tracker_Angle)
    # return pygame.surfarray.array3d(observation).astype(np.uint8)
    return Canvas

def Draw_Sector(screen, color, Tracker,Last_Tracker_Angle,thickness=2):
    '''
    绘制智能体可视区域（扇形）
    :param screen: 画面
    :param color: 区域边框颜色
    :param thickness: 边框宽度
    :return: None
    '''
    center = (Tracker['x'] + config.pixel_size // 2, Tracker['y'] + config.pixel_size // 2)  # 顶点

    axis_angle = Last_Tracker_Angle  # 对称轴方向
    sector_angle = config.max_detection_angle  # 张角
    radius = config.max_detection_distance  # 半径
    rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    start_angle = -(axis_angle - sector_angle)
    end_angle = -(axis_angle + sector_angle)
    start_angle2 = axis_angle - sector_angle
    end_angle2 = axis_angle + sector_angle
    pygame.draw.arc(screen, color, rect, math.radians(end_angle), math.radians(start_angle), thickness)  # 绘制弧形

    start_pos = (center[0] + radius * math.cos(math.radians(start_angle2)),
                 center[1] + radius * math.sin(math.radians(start_angle2)))
    end_pos = (center[0] + radius * math.cos(math.radians(end_angle2)),
               center[1] + radius * math.sin(math.radians(end_angle2)))
    pygame.draw.line(screen, color, center, start_pos, thickness)  # 绘制两条边
    pygame.draw.line(screen, color, center, end_pos, thickness)


def Get_Radar(Tracker,Angle,Static_Obstacles,Dynamic_Obstacles):
    '''
    获取16向雷达数据
    :return: to_avoid 激光雷达回传数据
    '''
    to_avoid = np.ones(16)  # 记录了探测范围内16个方向上是否有障碍物及其距离

    # 对地图边缘的探测
    for i in range(16):
        # 计算当前方向的角度（从智能体的视角）
        angle_rad = math.radians(i * 22.5)  # 每个方向22.5度
        direction_x = math.cos(angle_rad)
        direction_y = math.sin(angle_rad)

        # 计算理论上的最大距离（不考虑碰撞）
        max_dist_x = config.max_detection_distance * direction_x
        max_dist_y = config.max_detection_distance * direction_y

        # 检测与地图边缘的距离
        dist_to_left_edge = max(0, Tracker['x'] - 0)
        dist_to_right_edge = max(0, config.width - (Tracker['x'] + max_dist_x))
        dist_to_top_edge = max(0, Tracker['y'] - 0)
        dist_to_bottom_edge = max(0, config.height - (Tracker['y'] + max_dist_y))

        # 更新雷达数据，取最近的边缘距离
        distance = min(dist_to_left_edge, dist_to_right_edge, dist_to_top_edge, dist_to_bottom_edge)
        if distance > config.max_detection_distance:
            continue
        to_avoid[i] = distance / config.max_detection_distance

    # 对障碍的探测
    for obstacle in Static_Obstacles + Dynamic_Obstacles:
        # 计算障碍物相对位置
        dx = obstacle['x'] - Tracker['x']
        dy = obstacle['y'] - Tracker['y']
        # 计算障碍物与智能体的距离
        dist = np.sqrt(dx ** 2 + dy ** 2) - config.pixel_size / 2
        if dist > config.max_detection_distance:
            continue
        # 计算障碍物与智能体的相对角度
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        base_index = int(angle / 22.5)
        base_angle = Angle
        # 将角度映射到16个方向上
        basement = (base_index + 4) % 16  # 地图坐标系旋转
        direction_index = (basement - int(base_angle / 22.5)) % 16  # 相对角度
        # 更新雷达信息
        to_avoid[direction_index] = min(to_avoid[direction_index], dist / config.max_detection_distance)
    return to_avoid


def Move(action):
    '''
    获取智能体动作对应的坐标变化
    :param action:智能体动作
    :return: 局部坐标系下移动距离d和角度alpha
    '''
    # 0: 上, 1: 下, 2: 左, 3: 右, 4: 左上, 5: 右上, 6: 左下, 7: 右下, 8: 静止
    v = 5
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


def Map_Process():
    '''
    预置地图处理\n
    自由区域free_space：1-道路-灰色；2-活动场地-蓝色；3-草地-绿色\n
    障碍区域blocking_space:4-建筑物-棕色；5-固定障碍-橙色
    :return: None
    '''
    Free_space = []
    Map_Size = int(config.width / config.pixel_size)
    for row in range(0, Map_Size):
        for col in range(0, Map_Size):
            Free_space.append({'x': row * config.pixel_size, 'y': col * config.pixel_size})
    return Free_space

def Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,x, y, is_static=True, is_initialize=True):
    '''
    判断（x,y）位置是否空闲
    :param x: x坐标
    :param y: y坐标
    :param is_static:  非动态障碍物标志
    :param is_initialize: 初始化状态标志，仅在reset方法中使用
    :return:
    '''
    # 出界
    if x < 0 or x >= config.width or y < 0 or y >= config.height:
        return False
    # 障碍
    for obstacle in Static_Obstacles:
        if (abs(obstacle['x'] - x) <= config.pixel_size / 2 and abs(obstacle['y'] - y) <= config.pixel_size / 2):
            return False
    # target&dynamic_obstacle不能主动撞tracker
    if not is_initialize:
        if Tracker['x'] == x and Tracker['y'] == y:
            return False
    # 新增动态障碍物，要判断是否与动态障碍碰撞
    if not is_static:
        for obstacle in Dynamic_Obstacles:
            if abs(obstacle['x'] - x) <= 3 and abs(obstacle['y'] - y) <= 3:
                return False
    return True


def Get_Action_Mask(Tracker,Angle,Static_Obstacles,Dynamic_Obstacles):
    '''
    生成动作掩码
    :return: action_mask
    '''
    action_mask = np.ones(11)
    mask_dist = 0.3
    # 获取雷达信息
    radar = utils.Get_Radar(Tracker,Angle,Static_Obstacles,Dynamic_Obstacles)
    mask_flag = config.mask_flag
    # 根据雷达信息更新动作掩码
    if mask_flag:
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
    return action_mask


def Generate_Static_Obstacles(Tracker, Static_Obstacles, Dynamic_Obstacles,Blocking_Space):
    """
    随机均匀地生成静态障碍物\n
    地图分为25个区域，每个区域数量相同，区域内随机生成位置、大小、形状
    :return: static_obstacles
    """
    static_obstacles_density = 2  # 每个区域内障碍物数量
    static_obstacles = []
    obstacle_positions = utils.Generate_Random_Positions(static_obstacles_density)  # 生成随机位置

    # 根据位置随机生成障碍物形状和大小
    for position in obstacle_positions:
        shape = np.random.choice(['rectangle', 'cross', 'circle', 'parallel_line', 'vertical_line'],
                                 p=config.obstacle_kind_probability)
        if shape == 'rectangle':
            obstacle = utils.Generate_Rectangle(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space)
        elif shape == 'cross':
            obstacle = utils.Generate_Cross(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space)
        elif shape == 'circle':
            obstacle = utils.Generate_Circle(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space)
        elif shape == 'parallel_line':
            obstacle = utils.Generate_Parallel_Line(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space)
        else:
            obstacle = utils.Generate_Vertical_Line(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space)
        static_obstacles.extend(obstacle)
    return static_obstacles

def Generate_Static_Obstacles_Randomly(Tracker, Static_Obstacles, Dynamic_Obstacles,Blocking_Space):
    """
    随机不均匀地生成静态障碍物\n
    随机生成位置、大小、形状
    :return: static_obstacles
    """
    num_positions = 50  # 每个区域内障碍物数量
    static_obstacles = []
    positions = []  # 生成随机位置

    for _ in range(num_positions):
        a = np.random.randint(0, 49)
        b = np.random.randint(0, 49)
        x = a * config.pixel_size
        y = b * config.pixel_size
        if not ((x, y) in positions):
            positions.append((x, y))
    # 根据位置随机生成障碍物形状和大小
    for position in positions:
        shape = np.random.choice(['rectangle', 'cross', 'circle', 'parallel_line', 'vertical_line'],
                                 p=[0.4, 0.1, 0.1, 0.2, 0.2])
        if shape == 'rectangle':
            obstacle = utils.Generate_Rectangle(position, Tracker, Static_Obstacles, Dynamic_Obstacles,Blocking_Space)
        elif shape == 'cross':
            obstacle = utils.Generate_Cross(position, Tracker, Static_Obstacles, Dynamic_Obstacles,Blocking_Space)
        elif shape == 'circle':
            obstacle = utils.Generate_Circle(position, Tracker, Static_Obstacles, Dynamic_Obstacles,Blocking_Space)
        elif shape == 'parallel_line':
            obstacle = utils.Generate_Parallel_Line(position, Tracker, Static_Obstacles, Dynamic_Obstacles,Blocking_Space)
        else:
            obstacle = utils.Generate_Vertical_Line(position, Tracker, Static_Obstacles, Dynamic_Obstacles,Blocking_Space)
        static_obstacles.extend(obstacle)
    return static_obstacles



def Generate_Random_Positions(num_positions):
    """
    生成地图25个区域的随机位置
    :param num_positions: 每个区域所需位置数量
    :return: 位置坐标positions
    """
    positions = []
    area_dist = []
    for i in range(int(config.width / 100)):
        for j in range(int(config.width / 100)):
            area_dist.append({'x': i, 'y': j})
    for area in area_dist:
        for _ in range(num_positions):
            a = np.random.randint(0, 9)
            b = np.random.randint(0, 9)
            x = (config.pixel_size * area['x'] + a) * config.pixel_size
            y = (config.pixel_size * area['y'] + b) * config.pixel_size
            if not ((x, y) in positions):
                positions.append((x, y))
    return positions


def Generate_Rectangle(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space):
    """
    生成矩形障碍
    :param position:中心位置
    :return: 所有障碍覆盖坐标点obstacle
    """
    height = np.random.randint(config.min_obstacle_size // 2, config.max_obstacle_size // 2)
    width = np.random.randint(config.min_obstacle_size // 2, config.max_obstacle_size // 2)
    x, y = position
    obstacle = []
    for i in range(x - height // 2 * config.pixel_size, x + height // 2 * config.pixel_size, config.pixel_size):
        for j in range(y - width // 2 * config.pixel_size, y + width // 2 * config.pixel_size, config.pixel_size):
            if not({'x':i,'y':j} in Blocking_Space):
                if 0 <= i < config.width and 0 <= j < config.height and utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,i, j, is_static=True,is_initialize=True):
                    obstacle.append({"x": i, "y": j})

    return obstacle


def Generate_Cross(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space):
    """
    生成十字形障碍
    :param position:中心位置
    :return: 所有障碍覆盖坐标点obstacle
    """
    size = np.random.randint(config.min_obstacle_size, config.max_obstacle_size)
    thickness = np.random.randint(config.min_obstacle_size // 2, config.max_obstacle_size // 2)
    x, y = position
    obstacle = []
    for i in range(x - size // 2 * config.pixel_size, x + size // 2 * config.pixel_size, config.pixel_size):
        if 0 <= i < config.width and utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,i, y, is_static=True, is_initialize=True) and not({'x':i,'y':y} in Blocking_Space):
            obstacle.append({"x": i, "y": y})
    for j in range(y - size // 2 * config.pixel_size, y + size // 2 * config.pixel_size, config.pixel_size):
        if 0 <= j < config.height and utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,x, j, is_static=True, is_initialize=True) and not({'x':x,'y':j} in Blocking_Space):
            obstacle.append({"x": x, "y": j})
    return obstacle


def Generate_Circle(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space):
    '''
    生成圆形障碍
    :param position:中心位置
    :return: 所有障碍覆盖坐标点obstacle
    '''
    radius = np.random.randint(config.min_obstacle_size // 2, config.max_obstacle_size // 2)
    x, y = position
    obstacle = []
    for i in range(x - radius * config.pixel_size, x + (radius + 1) * config.pixel_size,config.pixel_size):
        for j in range(y - radius * config.pixel_size, y + (radius + 1) * config.pixel_size, config.pixel_size):
            if 0 <= i < config.width and 0 <= j < config.height and (x - i) ** 2 + (y - j) ** 2 <= (
                    radius * config.pixel_size) ** 2 and utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,i, j, is_static=True, is_initialize=True) and not({'x':i,'y':j} in Blocking_Space):
                obstacle.append({"x": i, "y": j})
    return obstacle


def Generate_Parallel_Line(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space):
    '''
    生成横向条形障碍
    :param position:中心位置
    :return: 所有障碍覆盖坐标点obstacle
    '''

    length = np.random.randint(config.min_obstacle_size, config.max_obstacle_size)
    x, y = position
    obstacle = []
    for i in range(x - length // 2 * config.pixel_size, x + length // 2 * config.pixel_size, config.pixel_size):
        if 0 <= i < config.width and utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,i, y, is_static=True, is_initialize=True) and not({'x':i,'y':y} in Blocking_Space):
            obstacle.append({"x": i, "y": y})
    return obstacle


def Generate_Vertical_Line(position,Tracker,Static_Obstacles,Dynamic_Obstacles,Blocking_Space):
    '''
    生成纵向条形障碍
    :param position:中心位置
    :return: 所有障碍覆盖坐标点obstacle
    '''
    length = np.random.randint(config.min_obstacle_size, config.max_obstacle_size)
    x, y = position
    obstacle = []
    for i in range(y - length // 2 * config.pixel_size, y + length // 2 * config.pixel_size, config.pixel_size):
        if 0 <= i < config.height and utils.Is_Free_Space(Tracker,Static_Obstacles,Dynamic_Obstacles,x, i, is_static=True, is_initialize=True) and not({'x':x,'y':i} in Blocking_Space):
            obstacle.append({"x": x, "y": i})
    return obstacle

