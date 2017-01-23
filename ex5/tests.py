from sol5 import *
import matplotlib.pyplot as plt
import sol5_utils


# model1, channels1 = learn_deblurring_model()
# model1.save_weights('blur_weights.h5')
# model1.save('blur_model.h5')

# model, channels = learn_denoising_model()
# model.save('denois_model.h5')
# model.save_weights('denois_weights.h5')


im1 = read_image(sol5_utils.images_for_deblurring()[13], 1)
cim = random_motion_blur(im1, [DEBLUR_KERNEL_SIZE])
mdl = build_nn_model(DEBLUR_PATCH_SIZE, DEBLUR_PATCH_SIZE, DEBLUR_CHANNELS)
mdl.load_weights('blur_weights.h5')
fix = restore_image(cim, mdl, DEBLUR_CHANNELS)

f = plt.figure()
f.add_subplot(121)
plt.imshow(cim, cmap=plt.cm.gray)
f.add_subplot(122)
plt.imshow(fix, cmap=plt.cm.gray)
plt.show()
