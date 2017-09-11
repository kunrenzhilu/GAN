from PIL import Image, ImageDraw
import os
import numpy as np
threshold = 245
dim = 64
line_dist = 8
row_num = dim // line_dist
width = 1
front_margin, back_margin = 2, 1
for fn in os.listdir("output/"):
    # rows = [[False for x in range(row_num)] for y in range(row_num + 1)]
    # cols = [[False for x in range(row_num)] for y in range(row_num + 1)]
    im = Image.open("output/" + fn)
    data = np.array(im)
    data = np.lib.pad(data, ((1, ), (1, ), (0, )), mode='constant', constant_values=255)
    im = Image.new("RGB", (dim, dim), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for y in range(1, dim + line_dist + 2, line_dist):
        for x in range(1, dim - line_dist + 2, line_dist):
            if np.mean(data[y-width:y+width+1, x+front_margin:x-back_margin+line_dist, 1]) <= threshold:
                # rows[y//line_dist][x//line_dist] = True
                draw.line([x-1, y-1, x+line_dist-1, y-1], fill=(255, 0, 0))
            if np.mean(data[x+front_margin:x-back_margin+line_dist, y-width:y+width+1, 1]) <= threshold:
                # cols[y//line_dist][x//line_dist] = True
                draw.line([y-1, x-1, y-1, x+line_dist-1], fill=(255, 0, 0))
    im.save("output_/" + fn)


