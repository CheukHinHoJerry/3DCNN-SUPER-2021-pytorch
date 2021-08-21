# -*- coding: utf-8 -*-
import numpy
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os.path
from pytorch.utils.load_data import *
import time

'''
resample – An optional resampling filter.
This can be one of PIL.Image.NEAREST (use nearest neighbour),
PIL.Image.BILINEAR (linear interpolation),
PIL.Image.BICUBIC (cubic spline interpolation),
or PIL.Image.LANCZOS (a high-quality downsampling filter).
If omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST.
'''


# getIndex is used to take the i average-splitted frames out -----------------------------------------------------------
def getIndex(AMOUNT, TOTAL_AMOUNT):
    prop = math.floor(
        TOTAL_AMOUNT / AMOUNT)  # because of math.floor, (i * prop) should be equal or smaller to TOTAL_AMOUNT

    all = range(TOTAL_AMOUNT)
    take = all[::prop]

    count = AMOUNT - len(take)
    appendIn = 2
    index = take
    while (count != 0):
        if count < 0:  # more pictures picked than asked, cut the latest few elements
            index = take[0:AMOUNT]
        if count > 0:  # less pictures picked than asked, append with the first few skipped elements
            take.append(appendIn)
            appendIn += 1
            index = take
        count = AMOUNT - len(index)

    # after while-loop return, should get exactly AMOUNT of picture index in the list "index"
    # print('index for selecting frames', index)
    return index


# pickFrame and pickCSV are used for single image datatype loading and processing ----------------------------------------------------------------------

# pickFrame works for data store as images  (shepp-logan-phantom.tif)
# The function will convert the image into YUV model, and take out the Y channel only
# The function will return a <non-type>; to use the output, wrap it with "numpy.array()"
def pickFrame(mode, i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, layer_3d):
    filename = 'frame' + str(i) + '.tif'

    # img = Image.open(path+'Frame_org_full/'+filename)
    # data = img.resize((TARGET_HEIGHT, TARGET_WIDTH))
    if mode == 'x':
        img = Image.open(path + 'Frame_cmp/' + filename)
        data = img.resize((HEIGHT, WIDTH))
        data = data.resize((TARGET_HEIGHT, TARGET_WIDTH), Image.BICUBIC)
    elif mode == 'y':
        img = Image.open(path + 'Frame_org_full/' + filename)
        data = img.resize((TARGET_HEIGHT, TARGET_WIDTH))

    data = numpy.array(data)

    if layer_3d == True and mode == 'y':
        # need to make y_set picture size smaller
        data = data[2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2]

    img_yuv = cv2.cvtColor(data, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    # returning y channel only for training and testing
    return y


# pack is the first step of multi-image processing -------------------------------------------------------------------
# The function will take "DEPTH" pictures out and pack as one small package
# The function will return a ndarray with a shape of (DEPTH, TARGET_HEIGHT, TARGET_WIDTH, CHANNEL)
def pack(DEPTH, mode, i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TOTAL_AMOUNT):
    HALF_RANGE = math.floor(DEPTH / 2)
    data = []
    if DEPTH % 2 == 0:
        RANGE = range(i - HALF_RANGE, i + HALF_RANGE)
    else:
        RANGE = range(i - HALF_RANGE, i + HALF_RANGE + 1)

    for j in RANGE:
        tmp = pickFrame(mode, j, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, True).tolist()
        # <class 'numpy.ndarray'> (64, 64)
        data.append(tmp)

    data = numpy.array(data)
    data = numpy.reshape(data, (DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1))
    return data  # return a single package


# packed_image_set is to pack all the packages into a dataset --------------------------------------------------------
def packed_image_set(DEPTH, AMOUNT, TOTAL_AMOUNT, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    x_set = []

    for i in getIndex(AMOUNT, TOTAL_AMOUNT):

        HALF_RANGE = math.floor(DEPTH / 2)

        if (i - HALF_RANGE) < 0 or (i + HALF_RANGE) >= TOTAL_AMOUNT:
            AMOUNT = AMOUNT - 1

        else:
            x_set.append(pack(DEPTH, 'x', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TOTAL_AMOUNT).tolist())

    x_set = numpy.reshape(numpy.array(x_set), (AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1))

    y_set = numpy.array([pickFrame('y', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, True) for i in
                         getIndex(AMOUNT, TOTAL_AMOUNT)])
    y_set = numpy.reshape(y_set, (AMOUNT, TARGET_HEIGHT - 2 * 2, TARGET_WIDTH - 2 * 2, 1))

    return (AMOUNT, x_set, y_set)


# single_image_set is for the final packing of single-image processing case --------------------------------------------------
# The function will reshape the final output into 4d tensor (the last dim stores color channel)
def single_image_set(AMOUNT, TOTAL_AMOUNT, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    x_set = numpy.array([pickFrame('x', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, False) for i in
                         getIndex(AMOUNT, TOTAL_AMOUNT)])
    x_set = numpy.reshape(x_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
    y_set = numpy.array([pickFrame('y', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, False) for i in
                         getIndex(AMOUNT, TOTAL_AMOUNT)])
    y_set = numpy.reshape(y_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
    return (x_set, y_set)


# this function returns numpy array containing compressed image (data_x) and original image (data_y), each 300
# when training with city data set, comment and comment necessary lines
def get_container_input(file_path):
    data_x = []
    data_y = []
    i = 0

    if not (os.path.isfile(file_path + "/Frame_org/frame" + str(0) + ".tif")):
        ValueError("No path found")

    while (os.path.isfile(file_path + "/Frame_org_full/frame" + str(i) + ".tif")):
        temp_org_img = Image.open(file_path + "/Frame_org_full/frame" + str(i) + ".tif")
        temp_cmp_img = Image.open(file_path + "/Frame_cmp/frame" + str(i) + ".tif")
        # temp_cmp_img = Image.open(file_path + "/Frame_blur_full/frame" + str(i) + ".tif")
        temp_org_array = numpy.array(temp_org_img)
        temp_cmp_array = numpy.array(temp_cmp_img)
        data_x.append(temp_cmp_array)
        data_y.append(temp_org_array)
        i += 1
    data_x, data_y = numpy.array(data_x), numpy.array(data_y)
    print("Shape of modified images: ", data_x.shape)
    print("Shape of original images: ", data_y.shape)

    return data_x, data_y


# for getting only y channel from the images (for training and testing)
def get_input_y_channel(file_path):
    data_x = []
    data_y = []
    i = 0

    if not (os.path.isfile(file_path + "/Frame_org/frame" + str(0) + ".tif")):
        ValueError("No path found")

    while (os.path.isfile(file_path + "/Frame_org/frame" + str(i) + ".tif")):
        temp_org_img = Image.open(file_path + "/Frame_org/frame" + str(i) + ".tif")
        # temp_cmp_img = Image.open(file_path+"/Frame_cmp/frame" + str(i) + ".tif")
        # temp_cmp_img = Image.open(file_path + "/Frame_blur/frame" + str(i) + ".tif")
        temp_cmp_img = Image.open(file_path + "/Frame_blur/frame" + str(i) + ".tif")
        temp_org_array = numpy.array(temp_org_img)
        temp_cmp_array = numpy.array(temp_cmp_img)
        temp_org_yuv_img = cv2.cvtColor(temp_org_array, cv2.COLOR_BGR2YUV)
        org_y, org_u, org_v = cv2.split(temp_org_yuv_img)
        temp_cmp_yuv_img = cv2.cvtColor(temp_cmp_array, cv2.COLOR_BGR2YUV)
        cmp_y, cmp_u, cmp_v = cv2.split(temp_cmp_yuv_img)
        (HEIGHT_ORG, WIDTH_ORG) = org_y.shape
        (HEIGHT_CMP, WIDTH_CMP) = cmp_y.shape
        # adding one dimension for channel
        org_y = org_y.reshape(HEIGHT_ORG, WIDTH_ORG, 1)
        cmp_y = cmp_y.reshape(HEIGHT_CMP, WIDTH_CMP, 1)

        data_x.append(cmp_y)
        data_y.append(org_y)
        i += 1
    data_x, data_y = numpy.array(data_x), numpy.array(data_y)
    print("Shape of modified images (only y channel): ", data_x.shape)
    print("Shape of original images: (only y channel)", data_y.shape)
    print(HEIGHT_CMP)
    return (data_x, data_y, HEIGHT_CMP, WIDTH_CMP, HEIGHT_ORG, WIDTH_ORG)


def get_input_y_channel_REDS_train(file_path):
    AMOUNT = 100

    for i in range(5):  # for each file
        for j in range(100):  # for each image, extract Y channel to return
            temp_org_img = Image.open(
                file_path + "/train/train_sharp/train/train_sharp/" + '/%03d' % i + "/%08d.png" % j)
            temp_cmp_img = Image.open(
                file_path + "/train/train_sharp_bicubic/train/train_sharp_bicubic/X4" + "/%03d/" % i + "%08d.png" % j)
            temp_org_array = numpy.array(temp_org_img)
            temp_cmp_array = numpy.array(temp_cmp_img)
            temp_org_yuv_img = cv2.cvtColor(temp_org_array, cv2.COLOR_BGR2YUV)
            org_y, org_u, org_v = cv2.split(temp_org_yuv_img)
            temp_cmp_yuv_img = cv2.cvtColor(temp_cmp_array, cv2.COLOR_BGR2YUV)
            cmp_y, cmp_u, cmp_v = cv2.split(temp_cmp_yuv_img)
            (HEIGHT_ORG, WIDTH_ORG) = org_y.shape
            (HEIGHT_CMP, WIDTH_CMP) = cmp_y.shape
            # adding one dimension for channel
            org_y = org_y.reshape(1, 1, HEIGHT_ORG, WIDTH_ORG)
            cmp_y = cmp_y.reshape(1, 1, HEIGHT_CMP, WIDTH_CMP)
            if j == 0:
                current_file_x = cmp_y
                current_file_y = org_y
            else:
                current_file_x = numpy.concatenate((current_file_x, cmp_y), axis=0)
                current_file_y = numpy.concatenate((current_file_y, org_y), axis=0)
        current_repacked_x = repacking(current_file_x)
        current_file_y = current_file_y[2:AMOUNT - 2, :, 2:HEIGHT_ORG - 2, 2:WIDTH_ORG - 2]
        if i == 0:
            data_x = current_repacked_x
            data_y = current_file_y
        else:
            data_x = numpy.concatenate((data_x, current_repacked_x), axis=0)
            data_y = numpy.concatenate((data_y, current_file_y), axis=0)

    data_x, data_y = numpy.array(data_x), numpy.array(data_y)
    return (data_x, data_y, HEIGHT_CMP, WIDTH_CMP, HEIGHT_ORG, WIDTH_ORG)


def get_input_y_channel_REDS_valid(file_path):
    # amount of images for each file, for cutting the edge of the image
    VAL_AMOUNT = 100
    for i in range(1):  # for each file
        for j in range(100):  # for each image, extra Y channel to return
            temp_org_img = Image.open(file_path + "/valid/val_sharp/val/val_sharp" + "/%03d/" % i + "%08d.png" % j)
            temp_cmp_img = Image.open(
                file_path + "/valid/val_sharp_bicubic/val/val_sharp_bicubic/X4" + "/%03d/" % i + "%08d.png" % j)
            temp_org_array = numpy.array(temp_org_img)
            temp_cmp_array = numpy.array(temp_cmp_img)
            temp_org_yuv_img = cv2.cvtColor(temp_org_array, cv2.COLOR_BGR2YUV)
            org_y, org_u, org_v = cv2.split(temp_org_yuv_img)
            temp_cmp_yuv_img = cv2.cvtColor(temp_cmp_array, cv2.COLOR_BGR2YUV)
            cmp_y, cmp_u, cmp_v = cv2.split(temp_cmp_yuv_img)
            (HEIGHT_ORG, WIDTH_ORG) = org_y.shape
            (HEIGHT_CMP, WIDTH_CMP) = cmp_y.shape
            # adding one dimension for channel
            org_y = org_y.reshape(1, 1, HEIGHT_ORG, WIDTH_ORG)
            cmp_y = cmp_y.reshape(1, 1, HEIGHT_CMP, WIDTH_CMP)
            if j == 0:
                current_file_x = cmp_y
                current_file_y = org_y
            else:
                current_file_x = numpy.concatenate((current_file_x, cmp_y), axis=0)
                current_file_y = numpy.concatenate((current_file_y, org_y), axis=0)
        current_repacked_x = repacking(current_file_x)
        current_file_y = current_file_y[2:VAL_AMOUNT - 2, :, 2:HEIGHT_ORG - 2, 2:WIDTH_ORG - 2]
        if i == 0:
            data_x = current_repacked_x
            data_y = current_file_y
        else:
            data_x = numpy.concatenate((data_x, current_repacked_x), axis=0)
            data_y = numpy.concatenate((data_y, current_file_y), axis=0)
        return (data_x, data_y, HEIGHT_CMP, WIDTH_CMP, HEIGHT_ORG, WIDTH_ORG)
