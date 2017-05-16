import os

from helpers.configuration import Configuration


def get_frames_delta(frame_1, frame_2):
    return abs(frame_1 - frame_2) + 1


def get_delta_distance(point_1, point_2):
    if point_1.x == point_2.x:
        return abs(point_1.y - point_2.y)
    if point_1.y == point_2.y:
        return abs(point_1.x - point_2.x)


def get_avg_speed(distance, time):
    return distance / time


def write_tile(file, tile, global_i):
    delta_distance = get_delta_distance(tile.point_1, tile.point_2)
    avg_speed = get_avg_speed(delta_distance, get_frames_delta(tile.frame_1, tile.frame_2))
    x = tile.point_1.x
    y = tile.point_1.y
    for i in range(get_frames_delta(tile.frame_1, tile.frame_2)):
        if tile.tile_type is config.TILE_TYPE.A:
            x += 0
            y += avg_speed
        elif tile.tile_type is config.TILE_TYPE.B:
            x += avg_speed
            y += 0
        elif tile.tile_type is config.TILE_TYPE.C:
            x += 0
            y -= avg_speed
        elif tile.tile_type is config.TILE_TYPE.D:
            x -= avg_speed
            y += 0
        elif tile.tile_type is config.TILE_TYPE.E:
            x += 0
            y += avg_speed
        global_i += 1
        file.write('{} {} {}\n'.format(global_i, x, y))

    return global_i

config = Configuration()

if not os.path.exists(os.path.join(config.PATH_LABELS, 'tiles')):
    meta = open(os.path.join(config.PATH_LABELS, 'tiles'), "w+")
    meta.close()

with open(os.path.join(config.PATH_LABELS, 'tiles'), "a") as f:
    global_i = 0

    global_i = write_tile(f, config.TILE_1_A, global_i)
    global_i = write_tile(f, config.TILE_1_B, global_i)
    global_i = write_tile(f, config.TILE_1_C, global_i)
    global_i = write_tile(f, config.TILE_1_D, global_i)
    global_i = write_tile(f, config.TILE_1_E, global_i)

    global_i = write_tile(f, config.TILE_2_A, global_i)
    global_i = write_tile(f, config.TILE_2_B, global_i)
    global_i = write_tile(f, config.TILE_2_C, global_i)
    global_i = write_tile(f, config.TILE_2_D, global_i)
    global_i = write_tile(f, config.TILE_2_E, global_i)

    global_i = write_tile(f, config.TILE_3_A, global_i)
    global_i = write_tile(f, config.TILE_3_B, global_i)
    global_i = write_tile(f, config.TILE_3_C, global_i)
    global_i = write_tile(f, config.TILE_3_D, global_i)
    global_i = write_tile(f, config.TILE_3_E, global_i)

    global_i = write_tile(f, config.TILE_4_A, global_i)
    global_i = write_tile(f, config.TILE_4_B, global_i)
    global_i = write_tile(f, config.TILE_4_C, global_i)
    global_i = write_tile(f, config.TILE_4_D, global_i)
    global_i = write_tile(f, config.TILE_4_E, global_i)

    global_i = write_tile(f, config.TILE_5_A, global_i)
    global_i = write_tile(f, config.TILE_5_B, global_i)
    global_i = write_tile(f, config.TILE_5_C, global_i)
    global_i = write_tile(f, config.TILE_5_D, global_i)
    global_i = write_tile(f, config.TILE_5_E, global_i)

    global_i = write_tile(f, config.TILE_6_A, global_i)
    global_i = write_tile(f, config.TILE_6_B, global_i)
    global_i = write_tile(f, config.TILE_6_C, global_i)
    global_i = write_tile(f, config.TILE_6_D, global_i)
    global_i = write_tile(f, config.TILE_6_E, global_i)

    global_i = write_tile(f, config.TILE_7_A, global_i)
    global_i = write_tile(f, config.TILE_7_B, global_i)
    global_i = write_tile(f, config.TILE_7_C, global_i)
    global_i = write_tile(f, config.TILE_7_D, global_i)
    global_i = write_tile(f, config.TILE_7_E, global_i)

    f.close()
