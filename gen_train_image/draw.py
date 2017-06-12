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
    if (x < start_point) or \
            (y < start_point):
        return 0
    else:
        return 1


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
    draw_to_centre(image1, init_x, init_y, index)
    distort()


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
                x, y = random.randint(canvas_width / 4, canvas_width * 3/4), \
                       random.randint(canvas_height / 4, canvas_height * 3/4)
                im = im.rotate(random.randint(-15, 15), resample=PIL.Image.BILINEAR, center=[x, y])
            else:
                data = np.array(im)
                if bool(random.getrandbits(1)):
                    r0, g0, b0 = rgb_red
                    r1, g1, b1 = random.randint(0, 128), random.randint(0, 255), random.randint(0, 255)
                else:
                    r0, g0, b0 = rgb_blk
                    r1, g1, b1 = random.randint(10, 255), random.randint(10, 255), random.randint(10, 255)

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

    generate()
