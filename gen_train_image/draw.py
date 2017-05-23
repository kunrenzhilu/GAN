"""
usage: this code is used to generate grid-like trajectories. Simply do 
python draw.py 100
will generate 100 jpeg files
"""

__author__ = "Kun"

from Tkinter import *
from PIL import Image
import random
import ImageDraw
import os
import sys
import time

# check whether the next step is out of scope
def check(x,y):
    if (x < start_point or x > canvas_height) or \
            (y < start_point or y > canvas_width):
        return 0
    else: 
        return 1

def checkered(canvas, line_distance, loop_num, draw):
    
       # vertical lines at an interval of "line_distance" pixel
    for x in range(start_point, canvas_width + 1, line_distance):
        canvas.create_line(x, start_point, x, canvas_height, fill="#476042")
        draw.line([x-start_point, start_point-start_point, x-start_point, canvas_height-start_point], rgb_blk)
       # horizontal lines at an interval of "line_distance" pixel
    for y in range(start_point, canvas_height + 1, line_distance):
        canvas.create_line(start_point, y, canvas_width, y, fill="#476042")
        draw.line([start_point-start_point, y-start_point, canvas_width-start_point, y-start_point], rgb_blk)

    init_x = line_dis * random.randint(0, row_num) + start_point
    init_y = line_dis * random.randint(0, row_num) + start_point
    p_x1 = init_x
    p_y1 = init_y
    p_x2 = p_x1
    p_y2 = p_y1
    for i in range(0,loop_num):
        while True:
            d_x = random.randint(-1, 1)
            d_y = random.randint(-1, 1)
            coin = random.randint(0, 1)
            if coin:
                p_x2 = p_x1 + line_dis*d_x
            else:
                p_y2 = p_y1 + line_dis*d_y
            if check(p_x2, p_y2):
                break

        canvas.create_line(p_x1, p_y1, p_x2, p_y2, fill=bold_color, width=bold_width)       
        draw.line([p_x1, p_y1, p_x2, p_y2], rgb_red, width=bold_width)
        p_x1 = p_x2
        p_y1 = p_y2

if __name__ == "__main__":        
    master = Tk()

    bold_width = 2
    bold_color = "#FF0000"
    #hextorgb()
    rgb_red = (255,0,0)
    rgb_blk = (0, 0, 0)
    random.seed(time.time())
    loop_num = random.randint(20, 70)
    start_point = 0
    line_dis = 8
    canvas_extra = 1
    canvas_width = 64 + start_point + canvas_extra
    canvas_height= 64 + start_point + canvas_extra
    row_num = canvas_height / line_dis
    w = Canvas(master,
                width=canvas_width,
                height=canvas_height)
    w.pack()
    
    image1 = Image.new("RGB", (canvas_width,canvas_height), (255,255,255))
    draw = ImageDraw.Draw(image1)
    for i in range(int(sys.argv[1])):
        image1 = Image.new("RGB", (canvas_width,canvas_height), (255,255,255))
        draw = ImageDraw.Draw(image1)
        # random work for abitrary times
	loop_num = random.randint(20,70)
        checkered(w, line_dis, loop_num, draw)
        image1.save("images/"+str(i)+".jpeg")
    #mainloop()
    #os.system("open image.bmp");
