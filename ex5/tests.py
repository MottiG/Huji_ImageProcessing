from sol5 import *

model1, channels1 = learn_deblurring_model(False)
model1.save('blure_model.h5')
model1.save_weights('blur_weights.h5')

model, channels = learn_denoising_model(False)
model.save('denois_model.h5')
model.save_weights('denois_weights.h5')

