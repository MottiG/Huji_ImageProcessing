from sol1 import *
from sol2 import *

im = read_image("ex1/tests/external/jerusalem.jpg", 1)

# r1 = DFT2(im)
# r1 = np.log(1 +np.abs(np.fft.fftshift(r1)))
#
r1 = blur_spatial(im, 25)
r2 = blur_fourier(im, 25)

f = plt.figure()
a = f.add_subplot(1,2,1)
a.title.set_text('blur_spatial')
plt.imshow(r1, cmap=plt.cm.gray)
b = f.add_subplot(1,2,2)
b.title.set_text('blur_fourier')
plt.imshow(r2, cmap=plt.cm.gray)
# f.add_subplot(1,3,3)
# plt.imshow(r2, cmap=plt.cm.gray)
plt.show()
