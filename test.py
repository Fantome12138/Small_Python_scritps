import numpy as np

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

w = 1920
h = 1080

min_offset_x = 0.4
min_offset_y = 0.4
scale_low = 1 - min(min_offset_x, min_offset_y)
scale_high = scale_low + 0.2  # 0.8

new_ar = w / h
scale = rand(scale_low, scale_high)
if new_ar < 1:
    nh = int(scale * h)
    nw = int(nh * new_ar)
else:
    nw = int(scale * w)
    nh = int(nw / new_ar)
print(nw, nh)

place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
print(place_x, place_y)
index = 0
for i in range(4):
    dx = place_x[index]
    dy = place_y[index]
    print(dx, dy)
    index += 1