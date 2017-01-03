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


# im = read_image("im2.png", 1)
# pyr, ker = build_gaussian_pyramid(im, 20, 3)
# display_pyramid(pyr, 7)

blend2 = read_image(relpath('c.jpg'), 2)
blend1 = read_image(relpath('aq.jpg'), 2)
mask = read_image(relpath('mask1.jpg'), 1)

mask = mask.astype(np.bool)
mask_res = np.zeros((512, 1024, 3), dtype=np.float32)

mask_res[:, :, 0] = pyramid_blending(blend1[:, :, 0], blend2[:, :, 0], mask, 6, 11, 7)
mask_res[:, :, 1] = pyramid_blending(blend1[:, :, 1], blend2[:, :, 1], mask, 6, 11, 7)
mask_res[:, :, 2] = pyramid_blending(blend1[:, :, 2], blend2[:, :, 2], mask, 6, 11, 7)


plt.imshow(mask_res,  cmap=plt.cm.gray)
plt.show()





# newIm = laplacian_to_image(pyr, ker, [0.0005,0.0005,0.0005,0.0000005,0.00000000005,0.0005,0.0005])
# plt.imshow(newIm, cmap=plt.cm.gray)
# plt.show()
