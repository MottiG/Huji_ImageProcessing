from sol3 import *

# # # # # # # # # # START OF COPY&PASTE FROM EX1 - THE PRESUBMIT USES THE READ_IMAGE FUNCTION # # # # # # # # # #
from scipy.misc import imread
from skimage.color import rgb2gray
GREYSCALE, COLOR, RGBDIM = 1, 2, 3
MAX_PIX_VAL = 255
def is_valid_args(filename: str, representation: int) -> bool:
    return (filename is not None) and (representation == 1 or representation == 2) and isinstance(filename, str)
def is_rgb(im: np.ndarray) -> bool: return im.ndim == RGBDIM
def read_image(filename: str, representation: int) -> np.ndarray:
    if not is_valid_args(filename, representation):
        raise Exception("Please provide valid filename and representation code")
    try: im = imread(filename)
    except OSError: raise Exception("Filename should be valid image filename")
    if is_rgb(im) and (representation == GREYSCALE): return rgb2gray(im).astype(np.float32)
    elif not is_rgb(im) and (representation == COLOR): raise Exception("Converting greyscale to RGB is not supported")
    return im.astype(np.float32) / MAX_PIX_VAL
    # # # # # # # # # # # # # # # # # # # # END OF COPY&PASTE FROM EX1 # # # # # # # # # # # # # # # # # # # #

# from PIL import Image
# col = Image.open("aqb.jpg")
# gray = col.convert('L')
# bw = gray.point(lambda x: 0 if x<20 else 255, '1')
# bw.save("mask1.jpg")



# im = read_image("im2.png", 1)
# pyr, ker = build_gaussian_pyramid(im, 20, 3)
# display_pyramid(pyr, 7)
#
blending_example2()



# newIm = laplacian_to_image(pyr, ker, [0.0005,0.0005,0.0005,0.0000005,0.00000000005,0.0005,0.0005])
# plt.imshow(newIm, cmap=plt.cm.gray)
# plt.show()
