from sol1 import *

# EQUALIZE TEST:
# res = histogram_equalize(read_image("tests/external/jerusalem.jpg", 1))
# plt.imshow(res[0], cmap=plt.cm.gray)
# plt.show()
# plt.plot(res[1])
# plt.plot(res[2],'r')
# plt.show()
#



# QUANTIZE TEST:
# res1 = quantize(read_image("tests/external/jerusalem.jpg", 2), 4, 5)
# res2 = quantize(read_image("tests/external/jerusalem.jpg", 1), 4, 5)
#
# f = plt.figure()
# f.add_subplot(2, 3, 1)
# plt.imshow(read_image("tests/external/jerusalem.jpg", 2), cmap=plt.cm.gray)
# f.add_subplot(2, 3, 2)
# plt.imshow(res1[0], cmap=plt.cm.gray)
# f.add_subplot(2, 3, 3)
# plt.plot(res1[1])
# f.add_subplot(2, 3, 4)
# plt.imshow(read_image("tests/external/jerusalem.jpg", 1), cmap=plt.cm.gray)
# f.add_subplot(2, 3, 5)
# plt.imshow(res2[0], cmap=plt.cm.gray)
# f.add_subplot(2, 3, 6)
# plt.plot(res2[1])
# plt.show()
#



# QUANTIZE_RGB TEST:
# res = quantize_rgb(read_image("tests/external/jerusalem.jpg", 2), 4, 5)
# f = plt.figure()
# f.add_subplot(1, 2, 1)
# plt.imshow(read_image("tests/external/jerusalem.jpg", 2))
# f.add_subplot(1, 2, 2)
# plt.imshow(res[0])
# plt.show()
# plt.plot(res[1])
# plt.show()