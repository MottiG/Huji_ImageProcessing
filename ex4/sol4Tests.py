from sol4 import *
import matplotlib.pyplot as plt

im = read_image('external/backyard1.jpg', 1)
pyr = build_gaussian_pyramid(im, 3, 3)[0]
pos, desc = find_features(pyr)
a=3
# pos = harris_corner_detector(im)
# points = spread_out_corners(im, 4, 4, 7)
# plt.imshow(im, cmap=plt.cm.gray)
# plt.scatter(points[:, 0], points[:, 1], edgecolors='r')
# plt.show()