from sol5 import *
import matplotlib.pyplot as plt
import sol5_utils
import numpy as np

# model1, channels1 = sol5.learn_deblurring_model()
# model1.save_weights('blur_weights.h5')
# model1.save('blure_model.h5')

# model, channels = learn_denoising_model()
# model.save('denois_model.h5')
# model.save_weights('denois_weights.h5')


im1 = read_image(sol5_utils.images_for_denoising()[3], 1)
cim = add_gaussian_noise(im1, 0.0, 0.2)
mdl = build_nn_model(DENOIS_PATCH_SIZE, DENOIS_PATCH_SIZE, DENOIS_CHANNELS)
mdl.load_weights('denois_weights.h5')
fix = restore_image(cim, mdl, DENOIS_CHANNELS)

f = plt.figure()
f.add_subplot(121)
plt.imshow(cim, cmap=plt.cm.gray)
f.add_subplot(122)
plt.imshow(fix, cmap=plt.cm.gray)
plt.show()
