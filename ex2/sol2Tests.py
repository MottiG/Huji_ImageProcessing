import matplotlib.pyplot as plt
from sol2 import *

im = read_image("tests/test/external/monkey.jpg", 1)

# r1 = DFT2(im)
# r1 = np.log(1 +np.abs(np.fft.fftshift(r1)))
#
r1 = DFT2(im)
r2 = IDFT2(r1)
f = plt.figure()
a = f.add_subplot(1,2,1)
a.title.set_text('real(shift(log(DFT2+1))')
plt.imshow(np.real(np.fft.fftshift(np.log(r1+1))), cmap=plt.cm.gray)
b = f.add_subplot(1,2,2)
b.title.set_text('IDFT2')
plt.imshow(np.real(r2), cmap=plt.cm.gray)
plt.show()
