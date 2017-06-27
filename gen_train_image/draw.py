from tkinter import *

import PIL
from PIL import Image, ImageDraw, ImageOps
import random
import os
import math
import numpy as np
import sys
import time
import copy


def check(x, y):
    if (x < start_point) or (x > canvas_width + start_point)\
            or (y < start_point) or (y > canvas_height + start_point):
        return FALSE
    else:
        return TRUE


def draw_quadrant(n_quadrants, n_steps, n_images):
    image_counter = 0
    while image_counter < n_images:
        moves = []
        rows = [[FALSE for x in range(row_num)] for y in range(row_num + 1)]
        cols = [[FALSE for x in range(row_num)] for y in range(row_num + 1)]
        quadrants = [FALSE for x in range(5)]
        p0_x, p0_y = line_dis * random.randint(0, row_num), line_dis * random.randint(0, row_num)
        start_quadrant = get_quadrant((p0_x, p0_y))
        quadrants_count = 1
        quadrants[start_quadrant] = TRUE
        steps_counter = 0
        while steps_counter < n_steps:
            flag_quadrant_check = FALSE
            flag_line_redundancy_check = FALSE
            for i in range(20):
                d = random.choice([-1, 1])
                coin = bool(random.getrandbits(1))
                if coin:
                    p1_x = p0_x + line_dis * d
                    p1_y = p0_y
                else:
                    p1_x = p0_x
                    p1_y = p0_y + line_dis * d

                flag_quadrant_check = check_quadrants((p1_x, p1_y), quadrants_count, n_quadrants, quadrants)
                flag_line_redundancy_check = check_line_redundancy((p1_x, p1_y), rows, cols, coin, d)
                if flag_line_redundancy_check and flag_quadrant_check:
                    break

            if flag_quadrant_check and flag_line_redundancy_check:
                p_q = get_quadrant((p1_x, p1_y))
                if quadrants[p_q] is FALSE:
                    quadrants_count += 1
                    quadrants[p_q] = TRUE
                if coin:
                    rows[p1_y//line_dis][(p1_x-max(d, 0))//line_dis] = TRUE
                else:
                    cols[p1_x//line_dis][(p1_y-max(d, 0))//line_dis] = TRUE
                moves.append([p0_x, p0_y, p1_x, p1_y])
                p0_x = p1_x
                p0_y = p1_y
                steps_counter += 1
            else:
                break

        if steps_counter == n_steps and quadrants.count(True) == n_quadrants:
            image_counter += 1
            im = Image.new("RGB", (64, 64), (255, 255, 255))
            draw = ImageDraw.Draw(im)
            for x in range(0, 65, 8):
                draw.line([x, 0, x, 64], rgb_blk)
            for y in range(0, 65, 8):
                draw.line([0, y, 64, y], rgb_blk)
            for i in range(n_steps):
                draw.line(moves[i], fill=rgb_red, width=bold_width)
            im.save("images/" + str(image_counter) + ".png")


def check_quadrants(p, q_count, n_q, qs):
    if check(p[0], p[1]) is FALSE:
        return FALSE

    if n_q < q_count:
        return FALSE

    p_q = get_quadrant(p)

    if qs[p_q]:
        return TRUE
    elif q_count < n_q:
        return TRUE
    else:
        return FALSE


def check_line_redundancy(p1, rows, cols, coin, d):
    if check(p1[0], p1[1]) is FALSE:
        return FALSE
    if coin:
        return rows[p1[1]//line_dis][p1[0]//line_dis - max(d, 0)] is not TRUE
    else:
        return cols[p1[0]//line_dis][p1[1]//line_dis - max(d, 0)] is not TRUE



def get_quadrant(p):
    mid_y = canvas_height // 2
    mid_x = canvas_width // 2
    if p[0] <= mid_x:
        if p[1] <= mid_y:
            return 2
        else:
            return 3
    else:
        if p[1] > mid_y:
            return 1
        else:
            return 4


def checkered(canvas, line_distance, loop_num, draw):
    # vertical lines at an interval of "line_distance" pixel
    for x in range(start_point, canvas_width + 1, line_distance):
        canvas.create_line(x, start_point, x, canvas_height, fill="#476042")
        draw.line([x - start_point, start_point - start_point, x - start_point, canvas_height - start_point], rgb_blk)
        # horizontal lines at an interval of "line_distance" pixel
    for y in range(start_point, canvas_height + 1, line_distance):
        canvas.create_line(start_point, y, canvas_width, y, fill="#476042")
        draw.line([start_point - start_point, y - start_point, canvas_width - start_point, y - start_point], rgb_blk)

    init_x = line_dis * random.randint(0, row_num) + start_point
    init_y = line_dis * random.randint(0, row_num) + start_point
    p_x1 = init_x
    p_y1 = init_y
    p_x2 = p_x1
    p_y2 = p_y1
    for i in range(0, loop_num):
        while True:
            d_x = random.randint(-1, 1)
            d_y = random.randint(-1, 1)
            coin = random.randint(0, 1)
            if coin:
                p_x2 = p_x1 + line_dis * d_x
            else:
                p_y2 = p_y1 + line_dis * d_y
            if check(p_x2, p_y2):
                break

        canvas.create_line(p_x1, p_y1, p_x2, p_y2, fill=bold_color, width=bold_width)
        draw.line([p_x1, p_y1, p_x2, p_y2], rgb_red, width=bold_width)
        p_x1 = p_x2
        p_y1 = p_y2


def generate():
    init_x = canvas_height / 2
    init_y = canvas_width / 2
    index = [[0 for x in range(math.ceil((row_num + 1) / 2))] for y in range(math.ceil((row_num + 1) / 2))]
    #draw_to_centre(image1, init_x, init_y, index)
    # distort()


# all the paths end at the centre, though they are in fact extended from the centre
# naming: x + y + last_direction + index_at_point (due to multiple choices) + a/b/c/d (4 sections of the image)
def draw_to_centre(image, init_x, init_y, index):
    next1_x = init_x - line_dis
    next1_y = init_y
    if check(next1_x, next1_y):
        i = index[int(init_x / line_dis)][int(init_y / line_dis)]
        index[int(init_x / line_dis)][int(init_y / line_dis)] += 1;
        prefix = "images/" + str(int(init_x)).zfill(2) + str(int(init_y)).zfill(2) + "x" + str(i)
        image.save(prefix + "a" + ".png")
        a1 = Image.open(prefix + "a" + ".png")
        draw1 = ImageDraw.Draw(a1)
        draw1.line([init_x, init_y, next1_x, next1_y], rgb_red, width=bold_width)
        a1.save(prefix + "a" + ".png")
        b1 = ImageOps.mirror(a1)
        b1.save(prefix + "b" + ".png")
        c1 = ImageOps.flip(a1)
        c1.save(prefix + "c" + ".png")
        d1 = ImageOps.flip(b1)
        d1.save(prefix + "d" + ".png")
        draw_to_centre(a1, next1_x, next1_y, index)

    next2_x = init_x
    next2_y = init_y - line_dis
    if check(next2_x, next2_y):
        i = index[int(init_x / line_dis)][int(init_y / line_dis)]
        index[int(init_x / line_dis)][int(init_y / line_dis)] += 1;
        prefix = "images/" + str(int(init_x)).zfill(2) + str(int(init_y)).zfill(2) + "y" + str(i)
        image.save(prefix + "a" + ".png")
        a2 = Image.open(prefix + "a" + ".png")
        draw2 = ImageDraw.Draw(a2)
        draw2.line([init_x, init_y, next2_x, next2_y], rgb_red, width=bold_width)
        a2.save(prefix + "a" + ".png")
        b2 = ImageOps.mirror(a2)
        b2.save(prefix + "b" + ".png")
        c2 = ImageOps.flip(a2)
        c2.save(prefix + "c" + ".png")
        d2 = ImageOps.flip(b2)
        d2.save(prefix + "d" + ".png")
        draw_to_centre(a2, next2_x, next2_y, index)


def distort():
    for fn in os.listdir("images/"):
        for i in range(0, 10):
            im = Image.open("images/" + fn)
            if bool(random.getrandbits(1)):
                x, y = random.randint(canvas_width / 4, canvas_width * 3 / 4), \
                       random.randint(canvas_height / 4, canvas_height * 3 / 4)
                im = im.rotate(random.randint(-15, 15), resample=PIL.Image.BILINEAR, center=[x, y])
            else:
                data = np.array(im)
                if bool(random.getrandbits(1)):
                    r0, g0, b0 = rgb_red
                    r1, g1, b1 = random.randint(225, 255), random.randint(0, 30), random.randint(0, 30)
                else:
                    r0, g0, b0 = rgb_blk
                    r1, g1, b1 = random.randint(0, 20), random.randint(0, 20), random.randint(0, 20)

                red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
                mask = (red == r0) & (green == g0) & (blue == b0)
                data[:, :, :3][mask] = [r1, g1, b1]
                im = Image.fromarray(data)

            im.save("images/d" + str(distort.counter) + ".png")
            distort.counter += 1


distort.counter = 0

if __name__ == "__main__":
    master = Tk()

    bold_width = 2
    bold_color = "#FF0000"
    rgb_red = (255, 0, 0)
    rgb_blk = (0, 0, 0)

    line_dis = 8
    start_point = 0
    canvas_extra = 1
    canvas_width = 64 + start_point
    canvas_height = 64 + start_point
    row_num = canvas_height // line_dis

    # w = Canvas(master,
    #            width=canvas_width,
    #            height=canvas_height)
    # w.pack()

    image1 = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(image1)
    for x in range(start_point, canvas_width + 1, line_dis):
        # w.create_line(x, start_point, x, canvas_height, fill="#476042")
        draw.line([x - start_point, start_point - start_point, x - start_point, canvas_height - start_point], rgb_blk)
    for y in range(start_point, canvas_height + 1, line_dis):
        # w.create_line(start_point, y, canvas_width, y, fill="#476042")
        draw.line([start_point - start_point, y - start_point, canvas_width - start_point, y - start_point], rgb_blk)

    #generate()
    draw_quadrant(2, 12, 50000)
