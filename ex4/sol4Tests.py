from sol4 import *


im1 = read_image('external/backyard1.jpg', 1)
im2 = read_image('external/backyard2.jpg', 1)
pyr1 = build_gaussian_pyramid(im1, 3, 3)[0]
pyr2 = build_gaussian_pyramid(im2, 3, 3)[0]
pos1, desc1 = find_features(pyr1)
pos2, desc2 = find_features(pyr2)
match_ind1, match_ind2 = match_features(desc1, desc2, 0.5)


a=3
# pos = harris_corner_detector(im)
# points = spread_out_corners(im, 4, 4, 7)
# plt.imshow(im, cmap=plt.cm.gray)
# plt.scatter(points[:, 0], points[:, 1], edgecolors='r')
# plt.show()