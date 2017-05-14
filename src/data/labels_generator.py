import os

from src.helpers.configuration import Configuration


def get_frames_delta(frame_1, frame_2):
    return abs(frame_1 - frame_2) + 1


def get_delta_distance(point_1, point_2):
    if point_1.X == point_2.X:
        return abs(point_1.Y - point_2.Y)
    if point_1.Y == point_2.Y:
        return abs(point_1.X - point_2.X)


def get_avg_speed(distance, time):
    return distance / time


config = Configuration()

if not os.path.exists(os.path.join(config.PATH_LABELS, '1_tile')):
    meta = open(os.path.join(config.PATH_LABELS, '1_tile'), "w+")
    meta.close()

with open(os.path.join(config.PATH_LABELS, '1_tile'), "a") as f:
    global_i = 0
    #  A
    delta_distance_a = get_delta_distance(config.TILE_1_A.point_1, config.TILE_1_A.point_2)
    avg_speed_a = get_avg_speed(delta_distance_a, get_frames_delta(config.TILE_1_A.frame_1, config.TILE_1_A.frame_2))
    x = config.TILE_1_A.point_1.x
    y = config.TILE_1_A.point_1.y
    for i in range(get_frames_delta(config.TILE_1_A.frame_1, config.TILE_1_A.frame_2)):
        x += 0
        y += avg_speed_a
        global_i = i
        f.write('{} {} {}\n'.format(i, x, y))

    # B
    delta_distance_b = get_delta_distance(config.TILE_1_B.point_1, config.TILE_1_B.point_2)
    avg_speed_b = get_avg_speed(delta_distance_b, get_frames_delta(config.TILE_1_B.frame_1, config.TILE_1_B.frame_2))
    x = config.TILE_1_B.point_1.x
    y = config.TILE_1_B.point_1.y
    for i in range(get_frames_delta(config.TILE_1_B.frame_1, config.TILE_1_B.frame_2)):
        x += avg_speed_b
        y += 0
        global_i += 1
        f.write('{} {} {}\n'.format(global_i, x, y))

    # C
    delta_distance_c = get_delta_distance(config.TILE_1_C.point_1, config.TILE_1_C.point_2)
    avg_speed_c = get_avg_speed(delta_distance_c, get_frames_delta(config.TILE_1_C.frame_1, config.TILE_1_C.frame_2))
    x = config.TILE_1_C.point_1.x
    y = config.TILE_1_C.point_1.y
    for i in range(get_frames_delta(config.TILE_1_C.frame_1, config.TILE_1_C.frame_2)):
        x += 0
        y -= avg_speed_c
        global_i += 1
        f.write('{} {} {}\n'.format(global_i, x, y))

    # D
    delta_distance_d = get_delta_distance(config.TILE_1_D.point_1, config.TILE_1_D.point_2)
    avg_speed_d = get_avg_speed(delta_distance_d, get_frames_delta(config.TILE_1_D.frame_1, config.TILE_1_D.frame_2))
    x = config.TILE_1_D.point_1.x
    y = config.TILE_1_D.point_1.y
    for i in range(get_frames_delta(config.TILE_1_D.frame_1, config.TILE_1_D.frame_2)):
        x -= avg_speed_d
        y += 0
        global_i += 1
        f.write('{} {} {}\n'.format(global_i, x, y))

    # E
    delta_distance_e = get_delta_distance(config.TILE_1_E.point_1, config.TILE_1_E.point_2)
    avg_speed_e = get_avg_speed(delta_distance_e, get_frames_delta(config.TILE_1_E.frame_1, config.TILE_1_E.frame_2))
    x = config.TILE_1_E.point_1.x
    y = config.TILE_1_E.point_1.y
    for i in range(get_frames_delta(config.TILE_1_E.frame_1, config.TILE_1_E.frame_2)):
        x += 0
        y += avg_speed_e
        global_i += 1
        f.write('{} {} {}\n'.format(global_i, x, y))

    f.close()
