import csv
from PIL import Image, ImageDraw
from os import remove

file_names = ["1.39_103.87.csv"]
lat_low, lat_high, lon_low, lon_high = 1.37, 1.41, 103.85, 103.89
lat_range = abs(lat_high - lat_low); lon_range = abs(lon_high - lon_low)
last_tuple = (-1, -1, -1) # (nid, doy, num)
counter = 0
im = Image.new("RGB", (64, 64), (255, 255, 255))
draw = ImageDraw.Draw(im)
for fn in file_names:
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if lat_low <= float(row['lat']) <= lat_high and lon_low <= float(row['lon']) <= lon_high:
                if (int(float(row['nid'])),\
                    int(float(row['day_of_year'])),\
                    int(float(row['label'])))\
                        != last_tuple:    # new trajectory
                    im.save("images/" + str(counter) + ".png")
                    counter += 1
                    if counter > 5000: break
                    im = Image.new("RGB", (64, 64), (255, 255, 255))
                    draw = ImageDraw.Draw(im)
                    last_tuple = (int(float(row['nid'])), \
                                  int(float(row['day_of_year'])), \
                                  int(float(row['label'])))
            draw.point([int(64 * (float(row['lat']) - lat_low) / lat_range),\
                       int(64 * (float(row['lon']) - lon_low) / lon_range)], fill=(255, 0, 0))

try:
    remove("images/0.png")
except:
    pass
