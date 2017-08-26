# Import libraries
import numpy as np  # linear algebra
from input import *
import tensorflow as tf


# Fast run length encoding
def rle(img, name):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
    flat_img[0] = 0
    flat_img[-1] = 0

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    s = name + ","
    for i in range(0, len(starts_ix)):
        if i != len(starts_ix) - 1:
            s = s + "%s %s " % (starts_ix[i], lengths[i])
        else:
            s = s + "%s %s" % (starts_ix[i], lengths[i])
    return s


def rle3(image, name):
   runs = rle_encode(image)
   rle_str = rle_to_string(runs)
   return name + "," + rle_str

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

# mask = np.array(Image.open('/home/yang/datasets/kaggle-carvana/data/train_masks/00087a6bd4dc_02_mask.gif'), dtype=np.uint8)
# idx, lengths = rle(mask)
#
# print len(idx)
# print len(lengths)


# files = get_train_image_files()

# directory = "/home/yang/datasets/kaggle-carvana/data/train_masks/"
#
# submission_file = open("./train_masks_test.csv", 'w')
# submission_file.writelines("img,rle_mask\n")
# for f in files:
#     print f
#     mask = np.array(Image.open(directory + f + "_mask.gif"), dtype=np.uint8)
#     starts_ix, lengths = rle(mask)
#     s = f + ".jpg" + ","
#     for i in range(0, len(starts_ix)):
#         if i != len(starts_ix) - 1:
#             s = s + "%s %s " % (starts_ix[i], lengths[i])
#         else:
#             s = s + "%s %s" % (starts_ix[i], lengths[i])
#     submission_file.writelines(s + "\n")
# submission_file.close()
