from sol5 import *
from sol5_utils import *

model, channels = learn_denoising_model(False)
model.save_weights('donis_weights.h5')