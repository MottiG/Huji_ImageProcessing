from sol5 import *
from sol5_utils import *

imlist = images_for_denoising()
bs = 5
ls = load_dataset(imlist, bs, type, (4,4))
tup = next(ls)
